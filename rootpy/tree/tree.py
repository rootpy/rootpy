import sys
import re
import fnmatch

import ROOT

from ..types import Variable
from ..core import Object, snake_case_methods, RequireFile
from ..plotting.core import Plottable
from ..plotting import Hist, Canvas
from ..registry import register
from ..utils import asrootpy
from .. import rootpy_globals as _globals
from .treeobject import TreeCollection, TreeObject
from .cut import Cut
from .buffer import TreeBuffer
from .model import TreeModel


class UserData(object):
    pass


@snake_case_methods
@register()
class Tree(Object, Plottable, RequireFile, ROOT.TTree):
    """
    Inherits from TTree so all regular TTree methods are available
    but certain methods (i.e. Draw) have been overridden
    to improve usage in Python
    """
    draw_command = re.compile('^.+>>[\+]?(?P<name>[^(]+).*$')

    def __init__(self, name=None,
                       title=None,
                       model=None,
                       file=None,
                       ignore_unsupported=False):

        if file:
            file.cd()
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

        if type(branches) not in (list, tuple):
            raise TypeError("branches must be a list or tuple")
        self._always_read = branches

    def use_cache(self, cache=True, cache_size=10000000, learn_entries=1):

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

        buffer = []
        for branch in self.iterbranches():
            if (Tree.branch_is_supported(branch) and
                self.GetBranchStatus(branch.GetName())):
                buffer.append((branch.GetName(), Tree.branch_type(branch)))
        return TreeBuffer(buffer, ignore_unsupported=self._ignore_unsupported)

    def create_branches(self, branches):

        if not isinstance(branches, TreeBuffer):
            branches = TreeBuffer(branches,
                                  ignore_unsupported=self._ignore_unsupported)
        self.set_buffer(branches, create_branches=True)

    def update_buffer(self, buffer, transfer_objects=False):

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

        return [branch for branch in self.GetListOfBranches()]

    def iterbranches(self):

        for branch in self.GetListOfBranches():
            yield branch

    @property
    def branchnames(self):

        return [branch.GetName() for branch in self.GetListOfBranches()]

    def iterbranchnames(self):

        for branch in self.iterbranches():
            yield branch.GetName()

    def glob(self, patterns, prune=None):
        """
        Return a list of branch names that match pattern.
        Exclude all matched branch names which also match a pattern in prune.
        prune may be a string or list of strings.
        """
        if isinstance(patterns, basestring):
            patterns = [patterns]
        if isinstance(prune, basestring):
            prune = [prune]
        matches = []
        for pattern in patterns:
            matches += fnmatch.filter(self.iterbranchnames(), pattern)
            if prune is not None:
                for prune_pattern in prune:
                    matches = [match for match in matches
                               if not fnmatch.fnmatch(match, prune_pattern)]
        return matches

    def __getitem__(self, item):

        if isinstance(item, basestring):
            return self.buffer[item]
        if not (0 <= item < len(self)):
            raise IndexError("entry index out of range")
        self.GetEntry(item)
        return self

    def GetEntry(self, entry):

        self.buffer.reset_collections()
        return ROOT.TTree.GetEntry(self, entry)

    def __iter__(self):

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
            i = 0
            while self.GetEntry(i):
                self.buffer._entry.set(i)
                yield self.buffer
                i += 1

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

        return self.GetEntries()

    def __contains__(self, branch):

        return self.has_branch(branch)

    def has_branch(self, branch):

        return not not self.GetBranch(branch)

    def csv(self, sep=',', branches=None,
            include_labels=True, limit=None,
            stream=None):
        """
        Print csv representation of tree only including branches
        of basic types (no objects, vectors, etc..)
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

        self.SetWeight(self.GetWeight() * value)

    def GetEntries(self, cut=None, weighted_cut=None, weighted=False):

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
            entries = ROOT.TTree.GetEntries(self, str(cut))
        else:
            entries = ROOT.TTree.GetEntries(self)
        if weighted:
            entries *= self.GetWeight()
        return entries

    def GetMaximum(self, expression, cut=None):

        if cut:
            self.Draw(expression, cut, "goff")
        else:
            self.Draw(expression, "", "goff")
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in xrange(min(n, 10000))]
        return max(vals)

    def GetMinimum(self, expression, cut=None):

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
        Convert selection (tree.Cut) to string
        """
        return super(Tree, self).CopyTree(str(selection), *args, **kwargs)

    def reset_branch_values(self):

        self.buffer.reset()

    def Fill(self, reset=False):

        super(Tree, self).Fill()
        # reset all branches
        if reset:
            self.buffer.reset()

    @RequireFile.cd
    def Write(self, *args, **kwargs):

        ROOT.TTree.Write(self, *args, **kwargs)

    def Draw(self,
             expression,
             selection="",
             options="",
             hist=None,
             min=None,
             max=None,
             bins=None,
             **kwargs):
        """
        Draw a TTree with a selection as usual,
        but return the created histogram.
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
        elif min is not None or max is not None:
            # handle graphics ourselves
            if options:
                options += ' '
            options += 'goff'
            if min is None:
                if max > 0:
                    min = 0
                else:
                    raise ValueError('must specify minimum')
            elif max is None:
                if min < 0:
                    max = 0
                else:
                    raise ValueError('must specify maximum')
            if bins is None:
                bins = 100
            local_hist = Hist(bins, min, max, **kwargs)
            expressions = ['%s>>+%s' % (expr, local_hist.GetName())
                           for expr in expressions]
        else:
            if 'goff' not in options:
                if not _globals.pad:
                    _globals.pad = Canvas()
                pad = _globals.pad
                pad.cd()
            match = re.match(Tree.draw_command, expression)
            histname = None
            if match:
                histname = match.group('name')
        for expr in expressions:
            ROOT.TTree.Draw(self, expr, selection, options)
        if hist is None and local_hist is None:
            if histname is not None:
                hist = asrootpy(ROOT.gDirectory.Get(histname))
            else:
                hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
            if hist:
                try:
                    hist.decorate(**kwargs)
                except:
                    pass
            if 'goff' not in options:
                pad.Modified()
                pad.Update()
            return hist
        elif local_hist is not None:
            local_hist.Draw(options)
            return local_hist
