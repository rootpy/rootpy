# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import, print_function

import sys
import re
import fnmatch

try:
    from collections import OrderedDict
except ImportError: # py 2.6
    from ..extern.ordereddict import OrderedDict

import ROOT

from .. import log; log = log[__name__]
from .. import asrootpy, QROOT
from .. import stl
from ..extern.shortuuid import uuid
from ..extern.six.moves import range
from ..extern.six import string_types
from ..context import set_directory, thread_specific_tmprootdir, do_nothing
from ..base import NamedObject
from ..decorators import snake_case_methods, method_file_check, method_file_cd
from ..plotting.base import Plottable
from ..plotting import Hist, Canvas
from ..memory.keepalive import keepalive
from .cut import Cut
from .treebuffer import TreeBuffer
from .treetypes import Scalar, Array, BaseChar
from .model import TreeModel


__all__ = [
    'Tree',
    'Ntuple',
]


class UserData(object):
    pass


class BaseTree(NamedObject):

    DRAW_PATTERN = re.compile(
        '^(?P<branches>.+?)'
        '(?P<redirect>\>\>[\+]?'
        '(?P<name>[^\(]+)'
        '(?P<binning>.+)?)?$')

    def _post_init(self):
        """
        The standard rootpy _post_init method that is used to initialize both
        new Trees and Trees retrieved from a File.
        """
        if not hasattr(self, '_buffer'):
            # only set _buffer if model was not specified in the __init__
            self._buffer = TreeBuffer()
        self.read_branches_on_demand = False
        self._branch_cache = {}
        self._current_entry = 0
        self._always_read = []
        self.userdata = UserData()
        self._inited = True

    def always_read(self, branches):
        """
        Always read these branches, even when in caching mode. Maybe you have
        caching enabled and there are branches you want to be updated for each
        entry even though you never access them directly. This is useful if you
        are iterating over an input tree and writing to an output tree sharing
        the same TreeBuffer and you want a direct copy of certain branches. If
        you have caching enabled but these branches are not specified here and
        never accessed then they will never be read from disk, so the values of
        branches in memory will remain unchanged.

        Parameters
        ----------
        branches : list, tuple
            these branches will always be read from disk for every GetEntry
        """
        if type(branches) not in (list, tuple):
            raise TypeError("branches must be a list or tuple")
        self._always_read = branches

    @classmethod
    def branch_type(cls, branch):
        """
        Return the string representation for the type of a branch
        """
        typename = branch.GetClassName()
        if not typename:
            leaf = branch.GetListOfLeaves()[0]
            typename = leaf.GetTypeName()
            # check if leaf has multiple elements
            leaf_count = leaf.GetLeafCount()
            if leaf_count:
                length = leaf_count.GetMaximum()
            else:
                length = leaf.GetLen()
            if length > 1:
                typename = '{0}[{1:d}]'.format(typename, length)
        return typename

    @classmethod
    def branch_is_supported(cls, branch):
        """
        Currently the branch must only have one leaf but the leaf may have one
        or multiple elements
        """
        return branch.GetNleaves() == 1

    def create_buffer(self, ignore_unsupported=False):
        """
        Create this tree's TreeBuffer
        """
        bufferdict = OrderedDict()
        for branch in self.iterbranches():
            # only include activated branches
            if not self.GetBranchStatus(branch.GetName()):
                continue
            if not BaseTree.branch_is_supported(branch):
                log.warning(
                    "ignore unsupported branch `{0}`".format(branch.GetName()))
                continue
            bufferdict[branch.GetName()] = Tree.branch_type(branch)
        self.set_buffer(TreeBuffer(
            bufferdict,
            ignore_unsupported=ignore_unsupported))

    def create_branches(self, branches):
        """
        Create branches from a TreeBuffer or dict mapping names to type names

        Parameters
        ----------
        branches : TreeBuffer or dict
        """
        if not isinstance(branches, TreeBuffer):
            branches = TreeBuffer(branches)
        self.set_buffer(branches, create_branches=True)

    def update_buffer(self, treebuffer, transfer_objects=False):
        """
        Merge items from a TreeBuffer into this Tree's TreeBuffer

        Parameters
        ----------
        buffer : rootpy.tree.buffer.TreeBuffer
            The TreeBuffer to merge into this Tree's buffer

        transfer_objects : bool, optional (default=False)
            If True then all objects and collections on the input buffer will
            be transferred to this Tree's buffer.
        """
        self._buffer.update(treebuffer)
        if transfer_objects:
            self._buffer.set_objects(treebuffer)

    def set_buffer(self, treebuffer,
                   branches=None,
                   ignore_branches=None,
                   create_branches=False,
                   visible=True,
                   ignore_missing=False,
                   ignore_duplicates=False,
                   transfer_objects=False):
        """
        Set the Tree buffer

        Parameters
        ----------
        treebuffer : rootpy.tree.buffer.TreeBuffer
            a TreeBuffer

        branches : list, optional (default=None)
            only include these branches from the TreeBuffer

        ignore_branches : list, optional (default=None)
            ignore these branches from the TreeBuffer

        create_branches : bool, optional (default=False)
            If True then the branches in the TreeBuffer should be created.
            Use this option if initializing the Tree. A ValueError is raised
            if an attempt is made to create a branch with the same name as one
            that already exists in the Tree. If False the addresses of existing
            branches will be set to point at the addresses in this buffer.

        visible : bool, optional (default=True)
            If True then the branches will be added to the buffer and will be
            accessible as attributes of the Tree.

        ignore_missing : bool, optional (default=False)
            If True then any branches in this buffer that do not exist in the
            Tree will be ignored, otherwise a ValueError will be raised. This
            option is only valid when ``create_branches`` is False.

        ignore_duplicates : bool, optional (default=False)
            If False then raise a ValueError if the tree already has a branch
            with the same name as an entry in the buffer. If True then skip
            branches that already exist. This option is only valid when
            ``create_branches`` is True.

        transfer_objects : bool, optional (default=False)
            If True, all tree objects and collections will be transferred from
            the buffer into this Tree's buffer.
        """
        # determine branches to keep while preserving branch order
        if branches is None:
            branches = treebuffer.keys()
        if ignore_branches is not None:
            branches = [b for b in branches if b not in ignore_branches]

        if create_branches:
            for name in branches:
                value = treebuffer[name]
                if self.has_branch(name):
                    if ignore_duplicates:
                        log.warning(
                            "Skipping entry in buffer with the same name "
                            "as an existing branch: `{0}`".format(name))
                        continue
                    raise ValueError(
                        "Attempting to create two branches "
                        "with the same name: `{0}`".format(name))
                if isinstance(value, Scalar):
                    self.Branch(name, value,
                        '{0}/{1}'.format(
                            name, value.type))
                elif isinstance(value, Array):
                    length = value.length_name or len(value)
                    self.Branch(name, value,
                        '{0}[{2}]/{1}'.format(
                            name, value.type, length))
                else:
                    self.Branch(name, value)
        else:
            for name in branches:
                value = treebuffer[name]
                if self.has_branch(name):
                    self.SetBranchAddress(name, value)
                elif not ignore_missing:
                    raise ValueError(
                        "Attempting to set address for "
                        "branch `{0}` which does not exist".format(name))
                else:
                    log.warning(
                        "Skipping entry in buffer for which no "
                        "corresponding branch in the "
                        "tree exists: `{0}`".format(name))
        if visible:
            newbuffer = TreeBuffer()
            for branch in branches:
                if branch in treebuffer:
                    newbuffer[branch] = treebuffer[branch]
            newbuffer.set_objects(treebuffer)
            self.update_buffer(newbuffer, transfer_objects=transfer_objects)

    def activate(self, branches, exclusive=False):
        """
        Activate branches

        Parameters
        ----------
        branches : str or list
            branch or list of branches to activate

        exclusive : bool, optional (default=False)
            if True deactivate the remaining branches
        """
        if exclusive:
            self.SetBranchStatus('*', 0)
        if isinstance(branches, string_types):
            branches = [branches]
        for branch in branches:
            if '*' in branch:
                matched_branches = self.glob(branch)
                for b in matched_branches:
                    self.SetBranchStatus(b, 1)
            elif self.has_branch(branch):
                self.SetBranchStatus(branch, 1)

    def deactivate(self, branches, exclusive=False):
        """
        Deactivate branches

        Parameters
        ----------
        branches : str or list
            branch or list of branches to deactivate

        exclusive : bool, optional (default=False)
            if True activate the remaining branches
        """
        if exclusive:
            self.SetBranchStatus('*', 1)
        if isinstance(branches, string_types):
            branches = [branches]
        for branch in branches:
            if '*' in branch:
                matched_branches = self.glob(branch)
                for b in matched_branches:
                    self.SetBranchStatus(b, 0)
            elif self.has_branch(branch):
                self.SetBranchStatus(branch, 0)

    @property
    def branches(self):
        """
        List of the branches
        """
        return [branch for branch in self.GetListOfBranches()]

    def iterbranches(self):
        """
        Iterator over the branches
        """
        for branch in self.GetListOfBranches():
            yield branch

    @property
    def branchnames(self):
        """
        List of branch names
        """
        return [branch.GetName() for branch in self.GetListOfBranches()]

    def iterbranchnames(self):
        """
        Iterator over the branch names
        """
        for branch in self.iterbranches():
            yield branch.GetName()

    def glob(self, patterns, exclude=None):
        """
        Return a list of branch names that match ``pattern``.
        Exclude all matched branch names which also match a pattern in
        ``exclude``. ``exclude`` may be a string or list of strings.

        Parameters
        ----------
        patterns: str or list
            branches are matched against this pattern or list of patterns where
            globbing is performed with '*'.

        exclude : str or list, optional (default=None)
            branches matching this pattern or list of patterns are excluded
            even if they match a pattern in ``patterns``.

        Returns
        -------
        matches : list
            List of matching branch names
        """
        if isinstance(patterns, string_types):
            patterns = [patterns]
        if isinstance(exclude, string_types):
            exclude = [exclude]
        matches = []
        for pattern in patterns:
            matches += fnmatch.filter(self.iterbranchnames(), pattern)
            if exclude is not None:
                for exclude_pattern in exclude:
                    matches = [match for match in matches
                               if not fnmatch.fnmatch(match, exclude_pattern)]
        return matches

    def __getitem__(self, item):
        """
        Get an entry in the tree or a branch

        Parameters
        ----------
        item : str or int
            if item is a str then return the value of the branch with that name
            if item is an int then call GetEntry
        """
        if isinstance(item, string_types):
            return self._buffer[item]
        self.GetEntry(item)
        return self

    def GetEntry(self, entry):
        """
        Get an entry. Tree collections are reset
        (see ``rootpy.tree.treeobject``)

        Parameters
        ----------
        entry : int
            entry index

        Returns
        -------
        ROOT.TTree.GetEntry : int
            The number of bytes read
        """
        if not (0 <= entry < self.GetEntries()):
            raise IndexError("entry index out of range: {0:d}".format(entry))
        self._buffer.reset_collections()
        return super(BaseTree, self).GetEntry(entry)

    def __iter__(self):
        """
        Iterator over the entries in the Tree.
        """
        if not self._buffer:
            self.create_buffer()
        if self.read_branches_on_demand:
            self._buffer.set_tree(self)
            # drop all branches from the cache
            self.DropBranchFromCache('*')
            for attr in self._always_read:
                try:
                    branch = self._branch_cache[attr]
                except KeyError:  # one-time hit
                    branch = self.GetBranch(attr)
                    if not branch:
                        raise AttributeError(
                            "branch `{0}` specified in "
                            "`always_read` does not exist".format(attr))
                    self._branch_cache[attr] = branch
                # add branches that we should always read to cache
                self.AddBranchToCache(branch)

            for i in range(self.GetEntries()):
                # Only increment current entry.
                # getattr on a branch will then GetEntry on only that branch
                # see ``TreeBuffer.get_with_read_if_cached``.
                self._current_entry = i
                self.LoadTree(i)
                for attr in self._always_read:
                    # Always read branched in ``self._always_read`` since
                    # these branches may never be getattr'd but the TreeBuffer
                    # should always be updated to reflect their current values.
                    # This is useful if you are iterating over an input tree
                    # and writing to an output tree that shares the same
                    # TreeBuffer but you don't getattr on all branches of the
                    # input tree in the logic that determines which entries
                    # to keep.
                    self._branch_cache[attr].GetEntry(i)
                self._buffer._entry.set(i)
                yield self._buffer
                self._buffer.next_entry()
                self._buffer.reset_collections()
        else:
            for i in range(self.GetEntries()):
                # Read all activated branches (can be slow!).
                super(BaseTree, self).GetEntry(i)
                self._buffer._entry.set(i)
                yield self._buffer
                self._buffer.reset_collections()

    def __setattr__(self, attr, value):
        if '_inited' not in self.__dict__ or attr in self.__dict__:
            return super(BaseTree, self).__setattr__(attr, value)
        try:
            return self._buffer.__setattr__(attr, value)
        except AttributeError:
            raise AttributeError(
                "`{0}` instance has no attribute `{1}`".format(
                    self.__class__.__name__, attr))

    def __getattr__(self, attr):
        if '_inited' not in self.__dict__:
            raise AttributeError(
                "`{0}` instance has no attribute `{1}`".format(
                    self.__class__.__name__, attr))
        try:
            return getattr(self._buffer, attr)
        except AttributeError:
            raise AttributeError(
                "`{0}` instance has no attribute `{1}`".format(
                    self.__class__.__name__, attr))

    def __setitem__(self, item, value):
        self._buffer[item] = value

    def __len__(self):
        """
        Same as GetEntries
        """
        return self.GetEntries()

    def __contains__(self, branch):
        """
        Same as has_branch
        """
        return self.has_branch(branch)

    def has_branch(self, branch):
        """
        Determine if this Tree contains a branch with the name ``branch``

        Parameters
        ----------
        branch : str
            branch name

        Returns
        -------
        has_branch : bool
            True if this Tree contains a branch with the name ``branch`` or
            False otherwise.
        """
        return not not self.GetBranch(branch)

    def csv(self, sep=',', branches=None,
            include_labels=True, limit=None,
            stream=None):
        """
        Print csv representation of tree only including branches
        of basic types (no objects, vectors, etc..)

        Parameters
        ----------
        sep : str, optional (default=',')
            The delimiter used to separate columns

        branches : list, optional (default=None)
            Only include these branches in the CSV output. If None, then all
            basic types will be included.

        include_labels : bool, optional (default=True)
            Include a first row of branch names labelling each column.

        limit : int, optional (default=None)
            Only include up to a maximum of ``limit`` rows in the CSV.

        stream : file, (default=None)
            Stream to write the CSV output on. By default the CSV will be
            written to ``sys.stdout``.
        """
        supported_types = (Scalar, Array, stl.string)
        if stream is None:
            stream = sys.stdout
        if not self._buffer:
            self.create_buffer(ignore_unsupported=True)
        if branches is None:
            branchdict = OrderedDict([
                (name, self._buffer[name])
                for name in self.iterbranchnames()
                if isinstance(self._buffer[name], supported_types)])
        else:
            branchdict = OrderedDict()
            for name in branches:
                if not isinstance(self._buffer[name], supported_types):
                    raise TypeError(
                        "selected branch `{0}` "
                        "is not a scalar or array type".format(name))
                branchdict[name] = self._buffer[name]
        if not branchdict:
            raise RuntimeError(
                "no branches selected or no "
                "branches of scalar or array types exist")
        if include_labels:
            # expand array types to f[0],f[1],f[2],...
            print(sep.join(
                name if isinstance(value, (Scalar, BaseChar, stl.string))
                    else sep.join('{0}[{1:d}]'.format(name, idx)
                                  for idx in range(len(value)))
                        for name, value in branchdict.items()),
                file=stream)
        # even though 'entry' is not used, enumerate or simply iterating over
        # self is required to update the buffer with the new branch values at
        # each tree entry.
        for i, entry in enumerate(self):
            line = []
            for value in branchdict.values():
                if isinstance(value, (Scalar, BaseChar)):
                    token = str(value.value)
                elif isinstance(value, stl.string):
                    token = str(value)
                else:
                    token = sep.join(map(str, value))
                line.append(token)
            print(sep.join(line), file=stream)
            if limit is not None and i + 1 == limit:
                break

    def Scale(self, value):
        """
        Scale the weight of the Tree by ``value``

        Parameters
        ----------
        value : int, float
            Scale the Tree weight by this value
        """
        self.SetWeight(self.GetWeight() * value)

    def GetEntries(self, cut=None, weighted_cut=None, weighted=False):
        """
        Get the number of (weighted) entries in the Tree

        Parameters
        ----------
        cut : str or rootpy.tree.cut.Cut, optional (default=None)
            Only entries passing this cut will be included in the count

        weighted_cut : str or rootpy.tree.cut.Cut, optional (default=None)
            Apply a weighted selection and determine the weighted number of
            entries.

        weighted : bool, optional (default=False)
            Multiply the number of (weighted) entries by the Tree weight.
        """
        if weighted_cut:
            hist = Hist(1, -1, 2)
            branch = self.GetListOfBranches()[0].GetName()
            weight = self.GetWeight()
            self.SetWeight(1)
            self.Draw('{0}=={1}>>{2}'.format(branch, branch, hist.GetName()),
                      weighted_cut * cut)
            self.SetWeight(weight)
            entries = hist.Integral()
        elif cut:
            entries = super(BaseTree, self).GetEntries(str(cut))
        else:
            entries = super(BaseTree, self).GetEntries()
        if weighted:
            entries *= self.GetWeight()
        return entries

    def GetMaximum(self, expression, cut=None):
        """
        TODO: we need a better way of determining the maximum value of an
        expression.
        """
        if cut:
            self.Draw(expression, cut, 'goff')
        else:
            self.Draw(expression, '', 'goff')
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in range(min(n, 10000))]
        return max(vals)

    def GetMinimum(self, expression, cut=None):
        """
        TODO: we need a better way of determining the minimum value of an
        expression.
        """
        if cut:
            self.Draw(expression, cut, "goff")
        else:
            self.Draw(expression, "", "goff")
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in range(min(n, 10000))]
        return min(vals)

    def CopyTree(self, selection, *args, **kwargs):
        """
        Copy the tree while supporting a rootpy.tree.cut.Cut selection in
        addition to a simple string.
        """
        return super(BaseTree, self).CopyTree(str(selection), *args, **kwargs)

    def reset_branch_values(self):
        """
        Reset all values in the buffer to their default values
        """
        self._buffer.reset()

    @method_file_cd
    def Write(self, *args, **kwargs):
        super(BaseTree, self).Write(*args, **kwargs)

    def Draw(self,
             expression,
             selection="",
             options="",
             hist=None,
             create_hist=False,
             **kwargs):
        """
        Draw a TTree with a selection as usual, but return the created
        histogram.

        Parameters
        ----------
        expression : str
            The expression to draw. Multidimensional expressions are separated
            by ":". rootpy reverses the expressions along each dimension so the
            order matches the order of the elements identifying a location in
            the resulting histogram. By default ROOT takes the expression "Y:X"
            to mean Y versus X but we argue that this is counterintuitive and
            that the order should be "X:Y" so that the expression along the
            first dimension identifies the location along the first axis, etc.

        selection : str or rootpy.tree.Cut, optional (default="")
            The cut expression. Only entries satisfying this selection are
            included in the filled histogram.

        options : str, optional (default="")
            Draw options passed to ROOT.TTree.Draw

        hist : ROOT.TH1, optional (default=None)
            The histogram to be filled. If not specified, rootpy will attempt
            to find what ROOT created and return that.

        create_hist : bool (default=False)
            If True and `hist`` is not specified and a histogram name is not
            specified in the draw expression, then override ROOT's
            default behaviour and fill a new histogram. ROOT will otherwise add
            points to a TGraph or TPolyMarker3D if not drawing in more than
            two dimensions.

        kwargs : dict, optional
            Remaining keword arguments are used to set the style attributes of
            the histogram.

        Returns
        -------
        If ``hist`` is specified, None is returned. If ``hist`` is left
        unspecified, an attempt is made to retrieve the generated histogram
        which is then returned.

        """
        if isinstance(expression, string_types):
            # Check that we have a valid draw expression
            exprmatch = re.match(BaseTree.DRAW_PATTERN, expression)
            if not exprmatch:
                raise ValueError(
                    "not a valid draw expression: `{0}`".format(expression))
            exprdict = exprmatch.groupdict()

            # Reverse variable order to match order in hist constructor
            fields = re.split('(?<!:):(?!:)', exprdict['branches'])
            num_dimensions = len(fields)
            expression = ':'.join(fields[:3][::-1] + fields[3:])
            if exprdict['redirect'] is not None:
                expression += exprdict['redirect']

        else:  # expression is list, tuple, ...
            fields = expression
            num_dimensions = len(fields)
            expression = ':'.join(fields[:3][::-1] + fields[3:])
            exprdict = {
                'branches': None,
                'redirect': None,
                'name': None,
                'binning': None,
                }

        if not isinstance(selection, Cut):
            # Let Cut handle any extra processing (i.e. ternary operators)
            selection = Cut(selection)

        graphics = 'goff' not in options

        if hist is not None:
            if not isinstance(hist, ROOT.TH1):
                raise TypeError("Cannot draw into a `{0}`".format(type(hist)))

            # Check that the dimensionality of the expression and object match
            if num_dimensions != hist.GetDimension():
                raise TypeError(
                    "The dimensionality of the expression `{0}` ({1:d}) "
                    "does not match the dimensionality of a `{2}`".format(
                        expression, num_dimensions, hist.__class__.__name__))
            # Handle graphics ourselves
            if graphics:
                if options:
                    options += ' '
                options += 'goff'
            if exprdict['name'] is None:
                # Draw into histogram supplied by user
                expression = '{0}>>+{1}'.format(expression, hist.GetName())
            else:
                if exprdict['name'] != hist.GetName():
                    # If the user specified a name to draw into then check that
                    # this is consistent with the specified object.
                    raise ValueError(
                        "The name specified in the draw "
                        "expression `{0}` does not match the "
                        "name of the specified object `{1}`".format(
                            exprdict['name'],
                            hist.GetName()))
                # Check that binning is not specified
                if exprdict['binning'] is not None:
                    raise ValueError(
                        "When specifying the object to draw into, do not "
                        "specify a binning in the draw expression")
        else:
            if create_hist and exprdict['name'] is None:
                if num_dimensions > 4:
                    raise ValueError(
                        "Cannot create a histogram for expressions with "
                        "more than 4 dimensions")
                newname = '{0}_{1}'.format(self.__class__.__name__, uuid())
                expression += '>>{0}'.format(newname)
                exprdict['name'] = newname

            pad = ROOT.gPad.func()
            own_pad = False

            if graphics and not pad:
                # Create a new canvas if one doesn't exist yet
                own_pad = True
                pad = Canvas()

        #  Note: TTree.Draw() pollutes gDirectory, make a temporary one
        with thread_specific_tmprootdir():
            if hist is not None:
                # If a custom histogram is specified (i.e, it's not being
                # created root side), then temporarily put it into the
                # temporary thread-specific directory.
                context = set_directory(hist)
            else:
                context = do_nothing()
            with context:
                super(BaseTree, self).Draw(expression, selection, options)

        if hist is None:
            # Retrieve histogram made by TTree.Draw
            if num_dimensions == 1 or exprdict['name'] is not None:
                # a TH1
                hist = asrootpy(self.GetHistogram(), warn=False)
            elif num_dimensions == 2:
                # a TGraph
                hist = asrootpy(pad.GetPrimitive('Graph'), warn=False)
            else:
                # ROOT: For a three and four dimensional Draw the TPolyMarker3D
                # is unnamed, and cannot be retrieved. Why, ROOT?
                log.warning(
                    "Cannot retrieve the TPolyMarker3D for "
                    "3D and 4D expressions")
                if graphics and own_pad:
                    # Since we cannot access the TPolyMarker3D we use self to
                    # keep the canvas alive
                    keepalive(self, pad)
            if hist: # is not None
                if isinstance(hist, Plottable):
                    hist.decorate(**kwargs)
                # ROOT, don't try to delete this object! (See issue #277)
                hist.SetBit(ROOT.kCanDelete, False)
                if graphics:
                    if own_pad:
                        # The usual bug is that the histogram is garbage
                        # collected and we want the canvas to keep the
                        # histogram alive, but here the canvas has been
                        # created locally and we are returning the histogram,
                        # so we want the histogram to keep the canvas alive.
                        keepalive(hist, pad)
                    # Redraw the histogram since we may have specified style
                    # attributes in **kwargs
                    hist.Draw()
            if graphics:
                pad.Modified()
                pad.Update()
        return hist

    def to_array(self, *args, **kwargs):
        """
        Convert this tree into a NumPy structured array
        """
        from root_numpy import tree2array
        return tree2array(self, *args, **kwargs)


@snake_case_methods
class Tree(BaseTree, QROOT.TTree):
    """
    Inherits from TTree so all regular TTree methods are available
    but certain methods (i.e. Draw) have been overridden
    to improve usage in Python.

    Parameters
    ----------
    name : str, optional (default=None)
        The Tree name (a UUID if None)

    title : str, optional (default=None)
        The Tree title (empty string if None)

    model : TreeModel, optional (default=None)
        If specified then this TreeModel will be used to create the branches
    """
    _ROOT = QROOT.TTree

    @method_file_check
    def __init__(self, name=None, title=None, model=None):
        super(Tree, self).__init__(name=name, title=title)
        self._buffer = TreeBuffer()
        if model is not None:
            if not issubclass(model, TreeModel):
                raise TypeError("the model must subclass TreeModel")
            self.set_buffer(model(), create_branches=True)
        self._post_init()

    def Fill(self, reset=False):
        """
        Fill the Tree with the current values in the buffer

        Parameters
        ----------
        reset : bool, optional (default=False)
            Reset the values in the buffer to their default values after
            filling.
        """
        super(Tree, self).Fill()
        # reset all branches
        if reset:
            self._buffer.reset()


@snake_case_methods
class Ntuple(BaseTree, QROOT.TNtuple):
    """
    Inherits from TNtuple so all regular TNtuple/TTree methods are available
    but certain methods (i.e. Draw) have been overridden
    to improve usage in Python.

    Parameters
    ----------
    varlist : list of str
        A list of the field names

    name : str, optional (default=None)
        The Ntuple name (a UUID if None)

    title : str, optional (default=None)
        The Ntuple title (empty string if None)

    bufsize : int, optional (default=32000)
        Basket buffer size
    """
    _ROOT = QROOT.TNtuple

    @method_file_check
    def __init__(self, varlist, name=None, title=None, bufsize=32000):
        super(Ntuple, self).__init__(':'.join(varlist), bufsize,
                                     name=name,
                                     title=title)
        self._post_init()
