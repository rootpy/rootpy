import sys
import re
import fnmatch

import ROOT

from ..types import Variable
from ..core import Object, snake_case_methods, RequireFile
from ..plotting.core import Plottable
from ..plotting import Hist, Canvas
from .. import log; log = log["__name__"]
from .. import asrootpy
from .. import rootpy_globals as _globals
from .treeobject import TreeCollection, TreeObject
from .cut import Cut
from .buffer import TreeBuffer
from .model import TreeModel


class UserData(object):
    pass


@snake_case_methods
class Tree(Object, Plottable, RequireFile, ROOT.TTree):
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

    ignore_unsupported : bool, optional (default=False)
        If True then branches of unsupported types will be ignored instead of
        raising a TypeError
    """
    DRAW_PATTERN = re.compile(
            '^(?P<branches>.+?)(?P<redirect>\>\>[\+]?(?P<name>[^\(]+).*)?$')

    def __init__(self, name=None,
                       title=None,
                       model=None,
                       ignore_unsupported=False):

        RequireFile.__init__(self)
        Object.__init__(self, name, title)
        self._ignore_unsupported = ignore_unsupported
        if model is not None:
            self.buffer = TreeBuffer(ignore_unsupported=ignore_unsupported)
            if not issubclass(model, TreeModel):
                raise TypeError("the model must subclass TreeModel")
            self.set_buffer(model(ignore_unsupported=ignore_unsupported),
                            create_branches=True)
        self._post_init(ignore_unsupported=ignore_unsupported)

    def _post_init(self, ignore_unsupported=False):
        """
        The standard rootpy _post_init method that is used to initialize both
        new Trees and Trees retrieved from a File.
        """
        self._ignore_unsupported = ignore_unsupported
        if not hasattr(self, "buffer"):
            self.buffer = TreeBuffer(
                ignore_unsupported=ignore_unsupported)
            self.set_buffer(self.create_buffer())
        Plottable.__init__(self)
        self._use_cache = False
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

    def use_cache(self, cache=True, cache_size=10000000, learn_entries=1):
        """
        Enable or disable the use of Tree caching

        Parameters
        ----------
        cache : bool, optional (default=True)
            enable or disable caching

        cache_size : int, optional (default=10000000)
            size of the cache in bytes

        learn_entries : int, optional (default=1)
            number of entries for the learning phase. ROOT will then read ahead
            more on the branches accessed most often during the learning phase.
        """
        if cache:
            self.buffer.set_tree(self)
            self.SetCacheSize(cache_size)
            ROOT.TTreeCache.SetLearnEntries(learn_entries)
        else:
            self.buffer.set_tree(None)
            # was the cache previously enabled?
            if self._use_cache:
                self.SetCacheSize(-1)
        self._use_cache = cache

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
            length = leaf.GetLen()
            if length > 1:
                typename = '%s[%d]' % (typename, length)
        return typename

    @classmethod
    def branch_is_supported(cls, branch):
        """
        Currently the branch must only have one leaf but the leaf may have one
        or multiple elements
        """
        return branch.GetNleaves() == 1

    def create_buffer(self):
        """
        Return a TreeBuffer for this Tree
        """
        buffer = {}
        for branch in self.iterbranches():
            if (Tree.branch_is_supported(branch) and
                self.GetBranchStatus(branch.GetName())):
                buffer[branch.GetName()] = Tree.branch_type(branch)
        return TreeBuffer(buffer, ignore_unsupported=self._ignore_unsupported)

    def create_branches(self, branches):
        """
        Create branches

        Paramaters
        ----------
        branches : list or dict
            Anything the TreeBuffer __init__ can handle, i.e. a list of 2-tuples
            of branch name and type or a dict mapping names to types.
        """
        if not isinstance(branches, TreeBuffer):
            branches = TreeBuffer(branches,
                                  ignore_unsupported=self._ignore_unsupported)
        self.set_buffer(branches, create_branches=True)

    def update_buffer(self, buffer, transfer_objects=False):
        """
        Merge items from a buffer into this Tree's buffer

        Parameters
        ----------
        buffer : rootpy.tree.buffer.TreeBuffer
            The TreeBuffer to merge into this Tree's buffer

        transfer_objects : bool, optional (default=False)
            If True then all objects and collections on the input buffer will be
            transferred to this Tree's buffer.
        """
        if self.buffer is not None:
            self.buffer.update(buffer)
            if transfer_objects:
                self.buffer.set_objects(buffer)
        else:
            self.buffer = buffer

    def set_buffer(self, buffer,
                   branches=None,
                   ignore_branches=None,
                   create_branches=False,
                   visible=True,
                   ignore_missing=False,
                   transfer_objects=False):
        """
        Set the Tree buffer

        Parameters
        ----------
        buffer : rootpy.tree.buffer.TreeBuffer
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
            If True and if create_branches is False then any branches in this
            buffer that do not exist in the Tree will be ignored, otherwise a
            ValueError will be raised.

        transfer_objects : bool, optional (default=False)
            If True, all tree objects and collections will be transferred from
            the buffer into this Tree's buffer.
        """
        # determine branches to keep
        all_branches = buffer.keys()
        if branches is None:
            branches = all_branches
        if ignore_branches is None:
            ignore_branches = []
        branches = (set(all_branches) & set(branches)) - set(ignore_branches)

        if create_branches:
            for name in branches:
                value = buffer[name]
                if self.has_branch(name):
                    raise ValueError(
                        "Attempting to create two branches "
                        "with the same name: %s" % name)
                if isinstance(value, Variable):
                    self.Branch(name, value, "%s/%s" % (name, value.type))
                else:
                    self.Branch(name, value)
        else:
            for name in branches:
                value = buffer[name]
                if self.has_branch(name):
                    self.SetBranchAddress(name, value)
                elif not ignore_missing:
                    raise ValueError(
                        "Attempting to set address for "
                        "branch %s which does not exist" % name)
        if visible:
            newbuffer = TreeBuffer(ignore_unsupported=self._ignore_unsupported)
            for branch in branches:
                if branch in buffer:
                    newbuffer[branch] = buffer[branch]
            newbuffer.set_objects(buffer)
            buffer = newbuffer
            self.update_buffer(buffer, transfer_objects=transfer_objects)

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
        if isinstance(branches, basestring):
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
        if isinstance(branches, basestring):
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
            branches matching this pattern or list of patterns are excluded even
            if they match a pattern in ``patterns``.

        Returns
        -------
        matches : list
            List of matching branch names
        """
        if isinstance(patterns, basestring):
            patterns = [patterns]
        if isinstance(exclude, basestring):
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
        if isinstance(item, basestring):
            return self.buffer[item]
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
            raise IndexError("entry index out of range: %d" % entry)
        self.buffer.reset_collections()
        return self.ROOT_base.GetEntry(self, entry)

    def __iter__(self):
        """
        Iterator over the entries in the Tree.
        """
        if self._use_cache:
            for i in xrange(self.GetEntries()):
                self._current_entry = i
                self.LoadTree(i)
                for attr in self._always_read:
                    try:
                        self._branch_cache[attr].GetEntry(i)
                    except KeyError:  # one-time hit
                        branch = self.GetBranch(attr)
                        if not branch:
                            raise AttributeError(
                                "branch %s specified in "
                                "'always_read' does not exist" % attr)
                        self._branch_cache[attr] = branch
                        branch.GetEntry(i)
                self.buffer._entry.set(i)
                yield self.buffer
                self.buffer.next_entry()
                self.buffer.reset_collections()
        else:
            for i in xrange(self.GetEntries()):
                self.ROOT_base.GetEntry(self, i)
                self.buffer._entry.set(i)
                yield self.buffer

    def __setattr__(self, attr, value):

        if '_inited' not in self.__dict__ or attr in self.__dict__:
            return super(Tree, self).__setattr__(attr, value)
        try:
            return self.buffer.__setattr__(attr, value)
        except AttributeError:
            raise AttributeError(
                "%s instance has no attribute '%s'" % \
                (self.__class__.__name__, attr))

    def __getattr__(self, attr):

        if '_inited' not in self.__dict__:
            raise AttributeError("%s instance has no attribute '%s'" % \
                                 (self.__class__.__name__, attr))
        try:
            return getattr(self.buffer, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % \
            (self.__class__.__name__, attr))

    def __setitem__(self, item, value):

        self.buffer[item] = value

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
        if stream is None:
            stream = sys.stdout
        if branches is None:
            branches = self.buffer.keys()
        branches = dict([(name, self.buffer[name]) for name in branches
                        if isinstance(self.buffer[name], Variable)])
        if not branches:
            return
        if include_labels:
            print >> stream, sep.join(branches.keys())
        # even though 'entry' is not used, enumerate or simply iterating over
        # self is required to update the buffer with the new branch values at
        # each tree entry.
        for i, entry in enumerate(self):
            print >> stream, sep.join([str(v.value) for v
                                       in branches.values()])
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
            self.Draw("%s==%s>>%s" % (branch, branch, hist.GetName()),
                      weighted_cut * cut)
            self.SetWeight(weight)
            entries = hist.Integral()
        elif cut:
            entries = self.ROOT_base.GetEntries(self, str(cut))
        else:
            entries = self.ROOT_base.GetEntries(self)
        if weighted:
            entries *= self.GetWeight()
        return entries

    def GetMaximum(self, expression, cut=None):
        """
        TODO: we need a better way of determining the maximum value of an
        expression.
        """
        if cut:
            self.Draw(expression, cut, "goff")
        else:
            self.Draw(expression, "", "goff")
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in xrange(min(n, 10000))]
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
        vals = [vals[i] for i in xrange(min(n, 10000))]
        return min(vals)

    def CopyTree(self, selection, *args, **kwargs):
        """
        Copy the tree while supporting a rootpy.tree.cut.Cut selection in
        addition to a simple string.
        """
        return super(Tree, self).CopyTree(str(selection), *args, **kwargs)

    def reset_branch_values(self):
        """
        Reset all values in the buffer to their default values
        """
        self.buffer.reset()

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
            self.buffer.reset()

    @RequireFile.cd
    def Write(self, *args, **kwargs):

        self.ROOT_base.Write(self, *args, **kwargs)

    def Draw(self,
             expression,
             selection="",
             options="",
             hist=None):
        """
        Draw a TTree with a selection as usual, but return the created
        histogram.

        Parameters
        ----------
        expression : str
            The expression or list of expression to draw. Multidimensional
            expressions are separated by ":". rootpy reverses the expressions
            along each dimension so the order matches the order of the elements
            identifying a location in the resulting histogram. By default ROOT
            takes the expression "Y:X" to mean Y versus X but we argue that
            this is counterintuitive and that the order should be "X:Y" so that
            the expression along the first dimension identifies the location
            along the first axis, etc.

        selection : str or rootpy.tree.Cut, optional (default="")
            The cut expression. Only entries satisfying this selection are
            included in the filled histogram.

        options : str, optional (default="")
            Draw options passed to ROOT.TTree.Draw

        hist: ROOT.TH1, optional (default=None)
            The histogram to be filled. If not specified ROOT will create one
            for you and rootpy will return it.

        Returns
        -------
        If ``hist`` is specified, None is returned. If ``hist`` is left
        unspecified, an attempt is made to retrieve the generated histogram
        which is then returned.
        """
        if isinstance(expression, (list, tuple)):
            expressions = expression
        else:
            expressions = [expression]
        if not isinstance(selection, Cut):
            # let Cut handle any extra processing (i.e. ternary operators)
            selection = Cut(selection)
        local_hist = None
        if hist is not None:
            # handle graphics ourselves
            if options:
                options += ' '
            options += 'goff'
            expressions = ['%s>>+%s' % (expr, hist.GetName())
                           for expr in expressions]
        else:
            if 'goff' not in options:
                if not _globals.pad:
                    _globals.pad = Canvas()
                pad = _globals.pad
                pad.cd()
            match = re.match(Tree.DRAW_PATTERN, expressions[0])
            histname = None
            if match and match.groupdict()['name']:
                histname = match.groupdict()['name']
        for expr in expressions:
            match = re.match(Tree.DRAW_PATTERN, expr)
            if not match:
                raise ValueError('not a valid draw expression: %s' % expr)
            # reverse variable order to match order in hist constructor
            groupdict = match.groupdict()
            expr = ':'.join(reversed(groupdict['branches'].split(':')))
            if groupdict['redirect']:
                expr += groupdict['redirect']
            self.ROOT_base.Draw(self, expr, selection, options)
        if hist is None and local_hist is None:
            if histname is not None:
                hist = asrootpy(ROOT.gDirectory.Get(histname))
            else:
                hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
            if hist:
                try:
                    hist.decorate(**kwargs)
                except:
                    exc_type, _, _ = sys.exc_info()
                    log.error("BUG: overly broad exception catch. "
                              "Please report this: '{0}'".format(exc_type))
                    pass
            if 'goff' not in options:
                pad.Modified()
                pad.Update()
            return hist
        elif local_hist is not None:
            local_hist.Draw(options)
            return local_hist

    def ndarray(self, branches=None,
                dtype=None,
                include_weight=False,
                weight_dtype='f4'):
        """
        Convert this tree into a NumPy ndarray
        """
        try:
            import numpy as np
            if dtype is None:
                dtype = np.float32
            from .. import root2array
            return root2array.tree_to_ndarray(self, branches,
                    dtype, include_weight, weight_dtype)
        except ImportError:
            raise ImportError('``ndarray`` requires NumPy')

    def recarray(self, branches=None,
                 include_weight=False,
                 weight_name='weight',
                 weight_dtype='f4'):
        """
        Convert this tree into a NumPy recarray
        """
        try:
            from .. import root2array
            return root2array.tree_to_recarray(self, branches,
                    include_weight, weight_name, weight_dtype)
        except ImportError:
            raise ImportError('``recarray`` requires NumPy')
