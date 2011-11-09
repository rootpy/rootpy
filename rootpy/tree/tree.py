import sys
import time
import re
import fnmatch
import types
import inspect
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
import ROOT
from ROOT import TTreeCache, gROOT
from ..types import *
from ..core import Object, camelCaseMethods, RequireFile
from ..plotting.core import Plottable
from ..registry import register, lookup_by_name, lookup_demotion
from ..utils import asrootpy, create
from ..io import open as ropen, DoesNotExist
from .filtering import *
from .treeobject import *


class TreeModelMeta(type):
    
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

        attrs = dict([(name + attr, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([name, cls.__name__]),
                    (TreeModel,), attrs)

    def suffix(cls, name):
        
        attrs = dict([(attr + name, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([cls.__name__, name]),
                    (TreeModel,), attrs)

    def get_attrs(cls):

        boring = dir(type('dummy', (object,), {})) + \
                 ['__metaclass__']
        attrs = [item for item in inspect.getmembers(cls)
                if item[0] not in boring
                and not isinstance(item[1], (types.FunctionType,
                                             types.MethodType,
                                             classmethod,
                                             staticmethod,
                                             property))]
        return attrs
        
    def to_struct(cls, name=None):
        """
        Convert model into a C struct then compile
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

    def __new__(cls):
        """
        Return a TreeBuffer for this TreeModel
        """
        buffer = TreeBuffer()
        for name, attr in cls.get_attrs():
            buffer[name] = attr()
        return buffer


@camelCaseMethods
@register()
class Tree(Object, Plottable, RequireFile, ROOT.TTree):
    """
    Inherits from TTree so all regular TTree methods are available
    but Draw has been overridden to improve usage in Python
    """
    draw_command = re.compile('^.+>>[\+]?(?P<name>[^(]+).*$')

    def __init__(self, name=None, title=None, model=None, file=None):
        
        if file:
            file.cd()
        RequireFile.__init__(self)
        Object.__init__(self, name, title)
        if model is not None:
            self.buffer = TreeBuffer()
            if not issubclass(model, TreeModel):
                raise TypeError("the model must subclass TreeModel")
            self.set_buffer(model(), create_branches=True)
        self._post_init()
    
    def _post_init(self):
        
        if not hasattr(self, "buffer"):
            self.buffer = TreeBuffer()
            self.set_buffer(self.create_buffer())
        Plottable.__init__(self)
        self._use_cache = False
        self._branch_cache = {}
        self._current_entry = 0
        self._always_read = []
        self._initialized = True

    def always_read(self, branches):
        
        if type(branches) not in (list, tuple):
            raise TypeError("branches must be a list or tuple")
        self._always_read = branches
    
    def use_cache(self, cache, cache_size=10000000, learn_entries=1):
        
        self._use_cache = cache
        if cache:
            self.buffer.set_tree(self)
            self.SetCacheSize(cache_size)
            TTreeCache.SetLearnEntries(learn_entries)
        else:
            self.buffer.set_tree(None)
            self.SetCacheSize(-1)

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
        return TreeBuffer(buffer)
    
    def create_branches(self, branches):

        if not isinstance(branches, TreeBuffer):
            branches = TreeBuffer(branches)
        self.set_branches_from_buffer(branches)
 
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
                yield self.buffer
                self.buffer.next_entry()
                self.buffer.reset_collections()
        else:
            i = 0
            while self.GetEntry(i):
                yield self.buffer
                i += 1
    
    def __setattr__(self, attr, value):
        
        if '_initialized' not in self.__dict__ or \
           attr in self.__dict__:
            return super(Tree, self).__setattr__(attr, value) 
        try:
            return self.buffer.__setattr__(attr, value)
        except AttributeError:
            raise AttributeError(
                "%s instance has no attribute '%s'" % \
                (self.__class__.__name__, attr))
    
    def __getattr__(self, attr):

        try:
            return getattr(self.buffer, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % \
            (self.__class__.__name__, attr))
    
    def update_buffer(self, buffer):

        if self.buffer is not None:
            self.buffer.update(buffer)
            self.buffer.set_objects(buffer)
        else:
            self.buffer = buffer

    def set_buffer(self, buffer,
                   variables=None,
                   create_branches=False,
                   visible=True,
                   ignore_missing=False,
                   transfer_object=True,
                   transfer_collections=True):
        
        if create_branches:
            for name, value in buffer.items():
                if variables is not None:
                    if name not in variables:
                        continue
                if self.has_branch(name):
                    raise ValueError(
                        "Attempting to create two branches "
                        "with the same name: %s" % name)
                if isinstance(value, Variable):
                    self.Branch(name, value, "%s/%s"% (name, value.type))
                else:
                    self.Branch(name, value)
        else:    
            for name, value in buffer.items():
                if variables is not None:
                    if name not in variables:
                        continue
                if self.has_branch(name):
                    self.SetBranchAddress(name, value)
                elif not ignore_missing:
                    raise ValueError(
                        "Attempting to set address for "
                        "branch %s which does not exist" % name)
        
        if visible:
            if variables:
                newbuffer = TreeBuffer()
                for variable in variables:
                    if variable in buffer:
                        newbuffer[variable] = buffer[variable]
                buffer = newbuffer
            self.update_buffer(buffer)
    
    def activate(self, variables, exclusive=False):

        if exclusive:
            self.SetBranchStatus('*', 0)
        if isinstance(variables, basestring):
            variables = [variables]
        for variable in variables:
            if self.has_branch(variable):
                self.SetBranchStatus(variable, 1)
    
    def deactivate(self, variables, exclusive=False):

        if exclusive:
            self.SetBranchStatus('*', 1)
        if isinstance(variables, basestring):
            variables = [variables]
        for variable in variables:
            if self.has_branch(variable):
                self.SetBranchStatus(variable, 0)

    def __getitem__(self, item):
        
        if isinstance(item, basestring):
            return self.buffer[item]
        if not (0 <= item < len(self)):
            raise IndexError("entry index out of range")
        self.GetEntry(item)
        return self
    
    def __setitem__(self, item, value):
        
        self.buffer[item] = value

    def __len__(self):

        return self.GetEntries()
    
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
    
    def glob(self, pattern, prune=None):
        """
        Return a list of branch names that match pattern.
        Exclude all matched branch names which also match a pattern in prune.
        prune may be a string or list of strings.
        """
        matches = fnmatch.filter(self.iterbranchnames(), pattern)
        if prune is not None:
            if isinstance(prune, basestring):
                prune = [prune]
            for prune_pattern in prune:
                matches = [match for match in matches
                           if not fnmatch.fnmatch(match, prune_pattern)]
        return matches
    
    def __contains__(self, branch):

        return self.has_branch(branch)
    
    def has_branch(self, branch):

        return not not self.GetBranch(branch)
    
    def csv(self, sep=',', include_labels=True, stream=sys.stdout):
        """
        Print csv representation of tree only including branches
        of basic types (no objects, vectors, etc..)
        """
        branches = dict([(name, value) for name, value in self.buffer.items()
                        if isinstance(value, Variable)])
        if not branches:
            return
        if include_labels:
            print >> stream, sep.join(branches.keys())
        for entry in self:
            print >> stream, sep.join([str(v.value) for v
                                       in branches.values()])
    
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
    
    def Draw(self, expression, selection="", options="", hist=None):
        """
        Draw a TTree with a selection as usual, but return the created histogram.
        """ 
        if hist is None:
            match = re.match(Tree.draw_command, expression)
            histname = None
            if match:
                histname = match.group('name')
                hist_exists = ROOT.gDirectory.Get(histname) is not None
        else:
            expression += ">>%s" % hist.GetName()
        ROOT.TTree.Draw(self, expression, selection, options)
        if hist is None:
            if histname is not None:
                hist = asrootpy(ROOT.gDirectory.Get(histname))
            else:
                hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
            return hist


class TreeChain(object):
    """
    A replacement for TChain
    """ 
    def __init__(self, name, files,
                 buffer=None,
                 branches=None,
                 events=-1,
                 stream=sys.stdout,
                 onfilechange=None,
                 cache=False,
                 cache_size=10000000,
                 learn_entries=1,
                 always_read=None):
        
        self.name = name
        if isinstance(files, tuple):
            files = list(files)
        elif not isinstance(files, list):
            files = [files]
        else:
            files = files[:]
        self.files = files
        self.buffer = buffer
        self.branches = branches
        self.weight = 1.
        self.tree = None
        self.file = None
        self.filters = EventFilterList()
        self.userdata = {}
        self.events = events
        self.total_events = 0
        self.initialized = False
        self.stream = stream
         
        if onfilechange is None:
            self.filechange_hooks = []
        else:
            self.filechange_hooks = onfilechange

        self.usecache = cache
        self.cache_size = cache_size
        self.learn_entries = learn_entries

        if not self.files:
            raise RuntimeError(
                "unable to initialize TreeChain: no files given")
        if not self.__rollover():
            raise RuntimeError("unable to initialize TreeChain")
        
        if always_read is None:
            self._always_read = []
        elif isinstance(always_read, basestring):
            if '*' in always_read:
                always_read = self.tree.glob(always_read)
            else:
                always_read = [always_read]
            self.always_read(always_read)
        else:
            branches = []
            for branch in always_read:
                if '*' in branch:
                    branchs += self.tree.glob(branch)
                else:
                    branchs.append(branch)
            self.always_read(branches)
    
    def always_read(self, branches):

        self._always_read = branches
        self.tree.always_read(branches)
     
    def __rollover(self):

        if self.tree is not None:
            self.tree = None
        if self.file is not None:
            self.file.Close()
            self.file = None
        if len(self.files) > 0:
            print >> self.stream, "%i file(s) remaining..." % len(self.files)
            fileName = self.files.pop()
            try:
                self.file = ropen(fileName)
            except IOError:
                self.file = None
                print >> self.stream, "WARNING: Skipping file. " \
                                      "Could not open file %s" % fileName
                return self.__rollover()
            try:
                self.tree = self.file.Get(self.name)
            except DoesNotExist:
                print >> self.stream, "WARNING: Skipping file. " \
                                      "Tree %s does not exist in file %s" % \
                                      (self.name, fileName)
                return self.__rollover()
            if len(self.tree.GetListOfBranches()) == 0:
                print >> self.stream, "WARNING: skipping tree with " \
                                      "no branches in file %s" % fileName
                return self.__rollover()
            if self.branches is not None:
                self.tree.activate(self.branches, exclusive=True)
            if self.buffer is None:
                self.buffer = self.tree.buffer
            else:
                self.tree.set_buffer(self.buffer, ignore_missing=True)
                self.tree.buffer.set_objects(self.buffer)
                self.buffer = self.tree.buffer
            self.tree.use_cache(self.usecache,
                                cache_size=self.cache_size,
                                learn_entries=self.learn_entries)
            self.tree.always_read(self._always_read)
            self.weight = self.tree.GetWeight()
            for target, args in self.filechange_hooks:
                target(*args, name=self.name, file=self.file)
            return True
        return False
    
    def __getattr__(self, attr):

        try:
            return getattr(self.tree, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % \
                (self.__class__.__name__, attr))

    def __getitem__(self, item):

        return self.tree.__getitem__(item)

    def __contains__(self, branch):
        
        return self.tree.__contains__(branch)
    
    def __iter__(self):
        
        passed_events = 0
        while True:
            t1 = time.time()
            entries = 0
            total_entries = float(self.tree.GetEntries())
            t2 = t1
            for entry in self.tree:
                entries += 1
                self.userdata = {}
                if self.filters(entry):
                    yield entry
                    passed_events += 1
                    if self.events == passed_events:
                        break
                if time.time() - t2 > 60:
                    print >> self.stream, \
                        "%i entries per second. %.0f%% done current tree." % \
                        (int(entries / (time.time() - t1)),
                        100 * entries / total_entries)
                    t2 = time.time()
            if self.events == passed_events:
                break
            print >> self.stream, "%i entries per second" % \
                int(entries / (time.time() - t1))
            print "Read %i bytes in %i transactions" % \
                (self.file.GetBytesRead(), self.file.GetReadCalls())
            self.total_events += entries
            if not self.__rollover():
                break


class TreeBuffer(dict):
    """
    A dictionary mapping variable names to values
    """
    def __init__(self, variables=None, tree=None):
        
        if variables is None:
            data = {}
        else:
            data = self.__process(variables)
        self._branch_cache = {}
        self._tree = tree
        self._current_entry = 0
        self._collections = {}
        self._objects = []
        super(TreeBuffer, self).__init__(data)
        self.userdata = {}
        self.__initialised = True

    def __process(self, variables):

        data = {}
        methods = dir(self)
        processed = []
        
        for name, vtype in variables:
            
            if name in methods or name.startswith('_'):
                raise ValueError("Illegal variable name: %s" % name)
            
            if name in processed:
                raise ValueError("Duplicate variable name %s" % name)
            else:
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
                raise TypeError("Unsupported variable type"
                                " for branch %s: %s" % (name, vtype))
            data[name] = obj
        return data
    
    def reset(self):
        
        for value in self.itervalues():
            if isinstance(value, Variable):
                value.reset()
            elif isinstance(value, ROOT.ObjectProxy):
                value.clear()
            else:
                value.__init__()

    def flat(self, variables=None):

        flat_variables = []
        if variables is None:
            variables = self.keys()
        for var in variables:
            demotion = lookup_demotion(self[var].__class__)
            if demotion is None:
                raise ValueError(
                    "Variable %s of type %s was not previously registered" % \
                    (var, self[var].__class__.__name__))
            flat_variables.append((var, demotion))
        return TreeBuffer(flat_variables)
    
    def update(self, variables):

        if not isinstance(variables, TreeBuffer):
            variables = self.__process(variables)
        dict.update(self, variables)

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
        Only if we are initialised
        """
        # this test allows attributes to be set in the __init__ method
        if not self.__dict__.has_key("_%s__initialised" % \
            self.__class__.__name__):
            return super(TreeBuffer, self).__setattr__(attr, value)
        elif self.__dict__.has_key(attr):
            # any normal attributes are handled normally
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
        for var, value in self.items():
            rep += "%s ==> %s\n" % (var, value)
        return rep
