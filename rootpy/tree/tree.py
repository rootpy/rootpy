import sys
import re
import fnmatch
import types
import inspect
from cStringIO import StringIO
import ROOT
from ROOT import TTreeCache, gROOT
from ..types import *
from ..core import Object, camelCaseMethods, RequireFile
from ..plotting.core import Plottable
from ..plotting import Hist
from ..registry import register, lookup_by_name, lookup_demotion
from ..utils import asrootpy, create
from .treeobject import *
from .cut import Cut


class TreeModelMeta(type):
    """
    Metaclass for all TreeModels
    Addition/subtraction of TreeModels is handled
    as set union and difference of class attributes
    """
    def __new__(cls, name, bases, dct):

        for attr, value in dct.items():
            TreeModelMeta.checkattr(attr, value)
        return type.__new__(cls, name, bases, dct)

    def __add__(cls, other):

        return type('_'.join([cls.__name__, other.__name__]),
                    (cls, other), {})

    def __iadd__(cls, other):

        return cls.__add__(other)

    def __sub__(cls, other):

        attrs = dict(set(cls.get_attrs()).difference(set(other.get_attrs())))
        return type('_'.join([cls.__name__, other.__name__]),
                    (TreeModel,), attrs)

    def __isub__(cls, other):

        return cls.__sub__(other)

    def __setattr__(cls, attr, value):

        TreeModelMeta.checkattr(attr, value)
        type.__setattr__(cls, attr, value)

    @classmethod
    def checkattr(metacls, attr, value):
        """
        Only allow class attributes that are instances of
        rootpy.types.Column, ROOT.TObject, or ROOT.ObjectProxy
        """
        if not isinstance(value, (types.MethodType,
                                  types.FunctionType,
                                  classmethod,
                                  staticmethod,
                                  property)):
            if attr in dir(type('dummy', (object,), {})) + \
                    ['__metaclass__']:
                return
            if attr.startswith('_'):
                raise SyntaxError("TreeModel attribute '%s' "
                                  "must not start with '_'" % attr)
            if not inspect.isclass(value):
                if not isinstance(value, Column):
                    raise TypeError("TreeModel attribute '%s' "
                                    "must be an instance of "
                                    "rootpy.types.Column" % attr)
                return
            if not issubclass(value, (ROOT.TObject, ROOT.ObjectProxy)):
                raise TypeError("TreeModel attribute '%s' must inherit "
                                "from ROOT.TObject or ROOT.ObjectProxy" % attr)

    def prefix(cls, name):
        """
        Create a new TreeModel where class attribute
        names are prefixed with name
        """
        attrs = dict([(name + attr, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([name, cls.__name__]),
                    (TreeModel,), attrs)

    def suffix(cls, name):
        """
        Create a new TreeModel where class attribute
        names are suffixed with name
        """
        attrs = dict([(attr + name, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([cls.__name__, name]),
                    (TreeModel,), attrs)

    def get_attrs(cls):
        """
        Get all class attributes
        """
        ignore = dir(type('dummy', (object,), {})) + \
                 ['__metaclass__']
        attrs = [item for item in inspect.getmembers(cls)
                if item[0] not in ignore
                and not isinstance(item[1], (types.FunctionType,
                                             types.MethodType,
                                             classmethod,
                                             staticmethod,
                                             property))]
        return attrs

    def to_struct(cls, name=None):
        """
        Convert TreeModel into a C struct then compile
        and import with ROOT
        """
        if name is None:
            name = cls.__name__
        basic_attrs = dict([(attr_name, value)
                            for attr_name, value in cls.get_attrs()
                            if isinstance(value, Column)])
        if not basic_attrs:
            return None
        src = 'struct %s {' % name
        for attr_name, value in basic_attrs.items():
            src += '%s %s;' % (value.type.typename, attr_name)
        src += '};'
        if gROOT.ProcessLine(src) != 0:
            return None
        try:
            exec 'from ROOT import %s; struct = %s' % (name, name)
            return struct
        except:
            return None

    def __repr__(cls):

        out = StringIO()
        for name, value in cls.get_attrs():
            print >> out, '%s -> %s' % (name, value)
        return out.getvalue()[:-1]

    def __str__(cls):

        return repr(cls)


class TreeModel(object):

    __metaclass__ = TreeModelMeta

    def __new__(cls, ignore_unsupported=False):
        """
        Return a TreeBuffer for this TreeModel
        """
        buffer = TreeBuffer(ignore_unsupported=ignore_unsupported)
        for name, attr in cls.get_attrs():
            buffer[name] = attr()
        return buffer


@camelCaseMethods
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
        self._inited = True

    def always_read(self, branches):

        if type(branches) not in (list, tuple):
            raise TypeError("branches must be a list or tuple")
        self._always_read = branches

    def use_cache(self, cache, cache_size=10000000, learn_entries=1):

        if cache:
            self.buffer.set_tree(self)
            self.SetCacheSize(cache_size)
            TTreeCache.SetLearnEntries(learn_entries)
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
            typename = branch.GetListOfLeaves()[0].GetTypeName()
        return typename

    def create_buffer(self):

        buffer = []
        for branch in self.iterbranches():
            if self.GetBranchStatus(branch.GetName()):
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
                    self.Branch(name, value, "%s/%s"% (name, value.type))
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
            if self.has_branch(branch):
                self.SetBranchStatus(branch, 1)

    def deactivate(self, branches, exclusive=False):

        if exclusive:
            self.SetBranchStatus('*', 1)
        if isinstance(branches, basestring):
            branches = [branches]
        for branch in branches:
            if self.has_branch(branch):
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
                    except KeyError: # one-time hit
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
            stream=sys.stdout):
        """
        Print csv representation of tree only including branches
        of basic types (no objects, vectors, etc..)
        """
        if branches is None:
            branches = self.buffer.keys()
        branches = dict([(name, self.buffer[name]) for name in branches
                        if isinstance(self.buffer[name], Variable)])
        if not branches:
            return
        if include_labels:
            print >> stream, sep.join(branches.keys())
        for i, entry in enumerate(self):
            print >> stream, sep.join([str(v.value) for v
                                       in branches.values()])
            if limit is not None and i + 1 == limit:
                break

    def Scale(self, value):

        self.SetWeight(self.GetWeight() * value)

    def GetEntries(self, cut = None, weighted_cut = None, weighted = False):

        if weighted_cut:
            hist = Hist(1,-1,2)
            branch = self.GetListOfBranches()[0].GetName()
            weight = self.GetWeight()
            self.SetWeight(1)
            self.Draw("%s==%s>>%s"%(branch, branch, hist.GetName()),
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

    def GetMaximum(self, expression, cut = None):

        if cut:
            self.Draw(expression, cut, "goff")
        else:
            self.Draw(expression, "", "goff")
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in xrange(min(n,10000))]
        return max(vals)

    def GetMinimum(self, expression, cut = None):

        if cut:
            self.Draw(expression, cut, "goff")
        else:
            self.Draw(expression, "", "goff")
        vals = self.GetV1()
        n = self.GetSelectedRows()
        vals = [vals[i] for i in xrange(min(n,10000))]
        return min(vals)

    def CopyTree(self, selection, *args, **kwargs):
        """
        Convert selection (tree.Cut) to string
        """
        return super(Tree, self).CopyTree(str(selection), *args, **kwargs)

    def reset(self):

        self.buffer.reset()

    def Fill(self, reset=False):

        super(Tree, self).Fill()
        # reset all branches
        if reset:
            self.buffer.reset()

    @RequireFile.cd
    def Write(self, *args, **kwargs):

        ROOT.TTree.Write(self, *args, **kwargs)

    def Draw(self, expression, selection="", options="",
                   hist=None,
                   min=None,
                   max=None,
                   bins=None,
                   **kwargs):
        """
        Draw a TTree with a selection as usual, but return the created histogram.
        """
        if isinstance(expression, (list, tuple)):
            expressions = expression
        else:
            expressions = [expression]
        if isinstance(selection, basestring) and selection:
            # let Cut handle any extra processing (i.e. ternary operators)
            selection = Cut(selection)
        local_hist = None
        if hist is not None:
            expressions = ['%s>>+%s' % (expr, hist.GetName())
                           for expr in expressions]
            # do not produce graphics if user specified histogram
            if options:
                options += ' '
            options += 'goff'
        elif min is not None or max is not None:
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
            match = re.match(Tree.draw_command, expression)
            histname = None
            if match:
                histname = match.group('name')
                hist_exists = ROOT.gDirectory.Get(histname) is not None
        for expr in expressions:
            ROOT.TTree.Draw(self, expr, selection, options)
        if hist is None and local_hist is None:
            if histname is not None:
                hist = asrootpy(ROOT.gDirectory.Get(histname))
            else:
                hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
            try: # bug (sometimes get a TObject)
                hist.decorate(**kwargs)
            except:
                pass
            return hist
        elif local_hist is not None:
            return local_hist


class TreeBuffer(dict):
    """
    A dictionary mapping branch names to values
    """
    def __init__(self, branches=None,
                       tree=None,
                       ignore_unsupported=False):

        super(TreeBuffer, self).__init__()
        self._fixed_names = {}
        self._branch_cache = {}
        self._tree = tree
        self._ignore_unsupported = ignore_unsupported
        self._current_entry = 0
        self._collections = {}
        self._objects = []
        self.userdata = {}
        self._entry = Int(0)
        if branches is not None:
            self.__process(branches)
        self._inited = True

    @classmethod
    def __clean(cls, branchname):

        # Replace invalid characters with '_'
        branchname = re.sub('[^0-9a-zA-Z_]', '_', branchname)
        # Remove leading characters until we find a letter or underscore
        return re.sub('^[^a-zA-Z_]+', '', branchname)

    def __process(self, branches):

        if not branches:
            return
        if not isinstance(branches, dict):
            try:
                branches = dict(branches)
            except TypeError:
                raise TypeError("branches must be a dict or anything "
                                "the dict constructor accepts")

        methods = dir(self)
        processed = []

        for name, vtype in branches.items():

            if name in processed:
                raise ValueError("duplicate branch name %s" % name)

            processed.append(name)

            # try to lookup type in registry
            cls, inits = lookup_by_name(vtype)
            obj = None
            if cls is not None:
                obj = cls()
                for init in inits:
                    init(obj)
            else:
                # last resort: try to create ROOT.'vtype'
                obj = create(vtype)
            if obj is None:
                if not self._ignore_unsupported:
                    raise TypeError("unsupported type "
                                    "for branch %s: %s" % (name, vtype))
            else:
                self[name] = obj

    def __setitem__(self, name, value):

        # for a key to be used as an attr it must be a valid Python identifier
        fixed_name = TreeBuffer.__clean(name)
        if fixed_name in dir(self) or fixed_name.startswith('_'):
            raise ValueError("illegal branch name: %s" % name)
        if fixed_name != name:
            self._fixed_names[fixed_name] = name
        super(TreeBuffer, self).__setitem__(name, value)

    def reset(self):

        for value in self.itervalues():
            if isinstance(value, Variable):
                value.reset()
            elif isinstance(value, ROOT.ObjectProxy):
                value.clear()
            else:
                value.__init__()

    def flat(self, branches=None):

        flat_branches = []
        if branches is None:
            branches = self.keys()
        for var in branches:
            demotion = lookup_demotion(self[var].__class__)
            if demotion is None:
                raise ValueError(
                    "branch %s of type %s was not previously registered" % \
                    (var, self[var].__class__.__name__))
            flat_branches.append((var, demotion))
        return TreeBuffer(flat_branches)

    def update(self, branches):

        if isinstance(branches, TreeBuffer):
            self._entry = branches._entry
            for name, value in branches.items():
                super(TreeBuffer, self).__setitem__(name, value)
            self._fixed_names.update(branches._fixed_names)
        else:
            self.__process(branches)

    def set_tree(self, tree=None):

        if tree is not None and not isinstance(tree, Tree):
            raise TypeError("tree must be a Tree instance or None")
        self._branch_cache = {}
        self._tree = tree
        self._current_entry = 0

    def next_entry(self):

        self._current_entry += 1

    def __setattr__(self, attr, value):
        """
        Maps attributes to values.
        Only if we are initialized
        """
        # this test allows attributes to be set in the __init__ method
        # any normal attributes are handled normally
        if '_inited' not in self.__dict__ or attr in self.__dict__:
            return super(TreeBuffer, self).__setattr__(attr, value)
        elif attr in self:
            variable = self.__getitem__(attr)
            if isinstance(variable, Variable):
                variable.set(value)
                return
            raise TypeError("cannot set non-Variable type "
                            "attribute '%s' of %s instance" % \
                            (attr, self.__class__.__name__))
        raise AttributeError("%s instance has no attribute '%s'" % \
                             (self.__class__.__name__, attr))

    def __getattr__(self, attr):

        if '_inited' not in self.__dict__:
            raise AttributeError("%s instance has no attribute '%s'" % \
                                 (self.__class__.__name__, attr))
        if attr in self._fixed_names:
            attr = self._fixed_names[attr]
        try:
            if self._tree is not None:
                try:
                    self._branch_cache[attr].GetEntry(self._current_entry)
                except KeyError: # one-time hit
                    branch = self._tree.GetBranch(attr)
                    if not branch:
                        raise AttributeError
                    self._branch_cache[attr] = branch
                    branch.GetEntry(self._current_entry)
            variable = self[attr]
            if isinstance(variable, Variable):
                return variable.value
            return variable
        except (KeyError, AttributeError):
            raise AttributeError("%s instance has no attribute '%s'" % \
                                 (self.__class__.__name__, attr))

    def reset_collections(self):

        for coll in self._collections.iterkeys():
            coll.reset()

    def define_collection(self, name, prefix, size, mix=None):

        coll = TreeCollection(self, name, prefix, size, mix=mix)
        object.__setattr__(self, name, coll)
        self._collections[coll] = (name, prefix, size, mix)

    def define_object(self, name, prefix, mix=None):

        cls = TreeObject
        if mix is not None:
            cls = mix_treeobject(mix)
        object.__setattr__(self, name, TreeObject(self, name, prefix))
        self._objects.append((name, prefix, mix))

    def set_objects(self, other):

        for args in other._objects:
            self.define_object(*args)
        for args in other._collections.itervalues():
            self.define_collection(*args)

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        rep = ""
        for name, value in self.items():
            rep += "%s ==> %s\n" % (name, value)
        return rep
