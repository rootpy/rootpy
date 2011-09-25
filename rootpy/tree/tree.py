import sys
import time
import re
import fnmatch
import ROOT
import inspect
from ..types import *
from ..core import Object, camelCaseMethods
from ..utils import *
from ..registry import register
from ..io import open as ropen
from .filtering import *
from ..plotting.core import Plottable
from .. import path

class TreeModel(object):

    @classmethod
    def get_user_attributes(cls):
        boring = dir(type('dummy', (object,), {}))+['get_user_attributes']
        return [item
                for item in inspect.getmembers(cls)
                if item[0] not in boring]


class TreeObject(object):

    def __init__(self, tree, name, prefix):

        self.tree = tree
        self.name = name
        self.prefix = prefix

    def __getitem__(self, thing):

        return getattr(self, thing)
         
    def __getattr__(self, attr):
        
        if attr.startswith(self.prefix):
            return getattr(self.tree, attr)
        return getattr(self.tree, self.prefix + attr)


class TreeCollectionObject(TreeObject):

    def __init__(self, tree, name, prefix, index):

        self.index = index
        super(TreeCollectionObject, self).__init__(tree, name, prefix)
         
    def __getattr__(self, attr):
        
        if attr.startswith(self.prefix):
            return getattr(self.tree, attr)[self.index]
        return getattr(self.tree, self.prefix + attr)[self.index]


__MIXINS__ = {}

def mix_treeobject(mixin):

    class TreeObject_mixin(TreeObject, mixin):
        
        def __init__(self, *args, **kwargs):

            TreeObject.__init__(self, *args, **kwargs)
            mixin.__init__(self)

    return TreeObject_mixin

def mix_treecollectionobject(mixin):

    class TreeCollectionObject_mixin(TreeCollectionObject, mixin):
        
        def __init__(self, *args, **kwargs):

            TreeCollectionObject.__init__(self, *args, **kwargs)
            mixin.__init__(self)

    return TreeCollectionObject_mixin


class TreeCollection(object):

    def __init__(self, tree, name, prefix, size, mixin=None):
        
        super(TreeCollection, self).__init__()
        self.tree = tree
        self.name = name
        self.prefix = prefix
        self.size = size
        
        self.tree_object_cls = TreeCollectionObject
        if mixin is not None:
            if mixin in __MIXINS__:
                self.tree_object_cls = __MIXINS__[mixin](tree, name, prefix, index)
            else:
                self.tree_object_cls = mix_treecollectionobject(mixin)
                __MIXINS__[mixin] = self.tree_object_cls
        
    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError(str(index))
        return self.tree_object_cls(self.tree, self.name, self.prefix, index)

    def __len__(self):

        return getattr(self.tree, self.size)
    
    def __iter__(self):

        for index in xrange(len(self)):
            yield self.tree_object_cls(self.tree, self.name, self.prefix, index)

@camelCaseMethods
@register
class Tree(Plottable, Object, ROOT.TTree):
    """
    Inherits from TTree so all regular TTree methods are available
    but Draw has been overridden to improve usage in Python
    """
    draw_command = re.compile('^.+>>[\+]?(?P<name>[^(]+).*$')

    def __init__(self, name=None, title=None, model=None):

        Object.__init__(self, name, title)
        Plottable.__init__(self)
        self.buffer = None
        if model is not None:
            if not issubclass(model, TreeModel):
                raise TypeError("the model must subclass TreeModel")
            attrs = model.get_user_attributes()
            buffer = TreeBuffer()
            for name, attr in attrs:
                buffer[name] = attr
            self.set_branches_from_buffer(buffer) 
        else:
            self.buffer = TreeBuffer()
        self.__initialised = True
    
    def _post_init(self):

        Plottable.__init__(self)
        self.build_buffer()
        self.__initialised = True

    def build_buffer(self):
        
        buffer = []
        for branch in self.iterbranches():
            if self.GetBranchStatus(branch.GetName()):
                typename = branch.GetClassName()
                if not typename:
                    typename = branch.GetListOfLeaves()[0].GetTypeName()
                buffer.append((branch.GetName(), typename))
        self.buffer = TreeBuffer(buffer)
    
    def create_branches(self, branches):

        if not isinstance(branches, TreeBuffer):
            branches = TreeBuffer(branches)
        self.set_branches_from_buffer(branches)
    
    def define_collection(self, name, prefix, size, mixin=None):
        
        super(Tree, self).__setattr__(name, TreeCollection(self, name, prefix, size, mixin=mixin))
    
    def define_object(self, name, prefix, mixin=None):

        cls = TreeObject
        if mixin is not None:
            cls = mix_treeobject(mixin) 
        super(Tree, self).__setattr__(name, TreeObject(self, name, prefix))

    def __getattr__(self, attr):

        try:
            return self.buffer.__getattr__(attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, attr))
    
    def __setattr__(self, attr, value):
        """
        Maps attributes to values.
        Only if we are initialised
        """
        # this test allows attributes to be set in the __init__ method
        if not self.__dict__.has_key("_%s__initialised" % self.__class__.__name__):
            return super(Tree, self).__setattr__(attr, value)
        elif self.__dict__.has_key(attr): # any normal attributes are handled normally
            return super(Tree, self).__setattr__(attr, value)
        else:
            try:
                return self.buffer.__setattr__(attr, value)
            except:
                raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, attr))
    
    def set_buffer(self, buffer):

        if self.buffer is not None:
            self.buffer.update(buffer)
        else:
            self.buffer = buffer

    def set_branches_from_buffer(self, buffer, variables = None, visible=True):
    
        for name, value in buffer.items():
            if variables is not None:
                if name not in variables:
                    continue
            if isinstance(value, Variable):
                self.Branch(name, value, "%s/%s"% (name, value.type))
            else:
                self.Branch(name, value)
        if visible:
            if variables:
                newbuffer = TreeBuffer()
                for variable in variables:
                    if variable in buffer:
                        newbuffer[variable] = buffer[variable]
                buffer = newbuffer
            self.set_buffer(buffer)

    def set_addresses_from_buffer(self, buffer, variables = None):
        
        for name, value in buffer.items():
            if variables is not None:
                if name not in variables:
                    continue
            if self.GetBranch(name):
                self.SetBranchAddress(name, value)
        self.set_buffer(buffer)
    
    def activate(self, variables, exclusive=False):

        if exclusive:
            self.SetBranchStatus('*', 0)
        if isinstance(variables, basestring):
            variables = [variables]
        for variable in variables:
            if self.GetBranch(variable):
                self.SetBranchStatus(variable, 1)
    
    def deactivate(self, variables, exclusive=False):

        if exclusive:
            self.SetBranchStatus('*', 1)
        if isinstance(variables, basestring):
            variables = [variables]
        for variable in variables:
            if self.GetBranch(variable):
                self.SetBranchStatus(variable, 0)

    def __getitem__(self, item):
        
        if isinstance(item, basestring):
            if self.buffer is not None:
                if self.buffer.has_key(item):
                    return self.buffer[item]
            if self.has_branch(item):
                return getattr(self, item)
            raise KeyError("%s"% item)
        if not (0 <= item < len(self)):
            raise IndexError("entry index out of range")
        self.GetEntry(item)
        return self

    def __len__(self):

        return self.GetEntries()
     
    def __iter__(self):

        i = 0
        while self.GetEntry(i):
            yield self
            i += 1
    
    def iterbranches(self):

        for branch in self.GetListOfBranches():
            yield branch
    
    def iterbranchnames(self):

        for branch in self.iterbranches():
            yield branch.GetName()
    
    def glob(self, pattern, *exclude):

        matches = fnmatch.filter(self.iterbranchnames(), pattern)
        for exc_pattern in exclude:
            matches = [match for match in matches if not fnmatch.fnmatch(match, exc_pattern)]
        return matches
    
    def has_branch(self, branch):

        return not not self.GetBranch(branch)
    
    def Scale(self, value):

        self.SetWeight(self.GetWeight() * value)
    
    def GetEntries(self, cut = None, weighted_cut = None, weighted = False):
        
        if weighted_cut:
            hist = Hist(1,-1,2)
            branch = self.GetListOfBranches()[0].GetName()
            weight = self.GetWeight()
            self.SetWeight(1)
            self.Draw("%s==%s>>%s"%(branch, branch, hist.GetName()), weighted_cut * cut)
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
                # if the hist already existed then I will
                # not overwrite its plottable features
                if not hist_exists and isinstance(hist, Plottable):
                    hist.decorate(self)
            else:
                hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
                if isinstance(hist, Plottable):
                    hist.decorate(self)
            return hist


class TreeChain(object):
    """
    A replacement for TChain
    """ 
    def __init__(self, name, files, buffer=None, branches=None, events=-1, stream=sys.stdout):
        
        self.name = name
        if isinstance(files, tuple):
            files = list(files)
        elif not isinstance(files, list):
            files = [files]
        else:
            files = files[:]
        self.files = path.expand_and_glob_all(files)
        self.buffer = buffer
        self.branches = branches
        self.weight = 1.
        self.tree = None
        self.file = None
        self.filters = EventFilterList()
        self.userdata = {}
        self.file_change_hooks = []
        self.events = events
        self.total_events = 0
        self.initialized = False
        self.stream = stream

    def init(self):

        if not self.files:
            raise RuntimeError("unable to initialize TreeChain: no files given")
        if not self.__initialize():
            raise RuntimeError("unable to initialize TreeChain")
        self.initialized = True
    
    def define_collection(self, name, prefix, size, mixin=None):
        
        setattr(self, name, TreeCollection(self, name, prefix, size, mixin=mixin))
    
    def define_object(self, name, prefix, mixin=None):

        cls = TreeObject
        if mixin is not None:
            cls = mix_treeobject(mixin) 
        setattr(self, name, TreeObject(self, name, prefix))
    
    def add_file_change_hook(self, target, args):
    
        self.file_change_hooks.append((target, args))

    def __initialize(self):

        if self.tree is not None:
            self.tree = None
        if self.file is not None:
            self.file.Close()
            self.file = None
        if len(self.files) > 0:
            if len(self.files) > 1:
                print >> self.stream, "%i files remaining..." % len(self.files)
            else:
                print >> self.stream, "1 file remaining..."
            fileName = self.files.pop()
            self.file = ropen(fileName)
            if not self.file:
                print >> self.stream, "WARNING: Skipping file. Could not open file %s"%(fileName)
                return self.__initialize()
            self.tree = self.file.Get(self.name)
            if not self.tree:
                print >> self.stream, "WARNING: Skipping file. Tree %s does not exist in file %s"%(self.name, fileName)
                return self.__initialize()
            if len(self.tree.GetListOfBranches()) == 0:
                # Try the next file:
                print >> self.stream, "WARNING: skipping tree with no branches in file %s"%fileName
                return self.__initialize()
            if self.branches is not None:
                self.tree.activate(self.branches, exclusive=True)
            if self.buffer is None:
                buffer = self.tree.buffer
                self.buffer = buffer
            self.tree.set_addresses_from_buffer(self.buffer)
            self.weight = self.tree.GetWeight()
            for target, args in self.file_change_hooks:
                target(*args, name=self.name, file=self.file)
            return True
        return False
    
    def __getattr__(self, attr):

        try:
            return getattr(self.tree, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, attr))

    def set_filters(self, filterlist):
        
        self.filters = filterlist

    def append_filter(self, filter):

        self.filters.append(filter)

    def prepend_filter(self, filter):

        self.filters.insert(0, filter)
    
    def __getitem__(self, item):

        return self.tree.__getitem__(item)

    def __iter__(self):
        
        if not self.initialized:
            self.init()
        passed_events = 0
        while True:
            t1 = time.time()
            entries = 0
            for entry in self.tree:
                entries += 1
                self.userdata = {}
                if self.filters(self):
                    yield self
                    passed_events += 1
                    if self.events == passed_events:
                        break
            if self.events == passed_events:
                break
            print "%i entries per second"% int(entries / (time.time() - t1))
            self.total_events += entries
            if not self.__initialize():
                break

class TreeBuffer(dict):
    """
    A dictionary mapping variable names to values
    """
    generate("vector<vector<float> >", "<vector>")
    generate("vector<vector<int> >", "<vector>")
    generate("vector<vector<unsigned int> >", "<vector>")
    generate("vector<vector<long> >", "<vector>")
    generate("vector<vector<unsigned long> >", "<vector>")
    generate("vector<vector<double> >", "<vector>")
    generate("vector<vector<string> >")

    demote = {"Bool_t": "B",
              "Float_t":"F",
              "Double_t": "D",
              "Int_t":"I",
              "UInt_t":"UI",
              "Int":"I",
              "Float":"F",
              "F":"F",
              "I":"I",
              "UI":"UI",
              "vector<float>":"F",
              "vector<int>":"I",
              "vector<unsigned int>":"UI",
              "vector<int, allocator<int> >":"I",
              "vector<float, allocator<float> >":"F",
              "VF":"F",
              "VI":"I",
              "VUI":"UI",
              "vector<vector<float> >":"VF",
              "vector<vector<int> >":"VI",
              "vector<vector<unsigned int> >":"VUI",
              "vector<vector<int>, allocator<vector<int> > >":"VI",
              "vector<vector<float>, allocator<vector<float> > >":"VF",
              "VVF":"VF",
              "VVI":"VI"} 

    def __init__(self, variables = None, default = -1111, flatten = False):
        
        self.variables = variables
        if self.variables is None:
            self.variables = []
            data = {}
        else:
            data = self.__process(self.variables, default, flatten)
        super(TreeBuffer, self).__init__(data)
        self.__initialised = True

    def __process(self, variables, default = -1111, flatten = False):

        data = {}
        methods = dir(self)
        processed = []
        for name, vtype in variables:
            if flatten:
                vtype = TreeBuffer.demote[vtype] #.upper()]
            if name in processed:
                raise ValueError("Duplicate variable name %s"%name)
            else:
                processed.append(name)
            if vtype.upper() in ("B", "BOOL_T"):
                data[name] = Bool(False)
            elif vtype.upper() in ("C", "CHAR_T"):
                data[name] = Char('\x00')
            elif vtype.upper() in ("I", "INT_T"):
                data[name] = Int(default)
            elif vtype.upper() in ("UI", "UINT_T"):
                data[name] = UInt(default)
            elif vtype.upper() in ("F", "FLOAT_T"):
                data[name] = Float(default)
            elif vtype.upper() in ("D", "DOUBLE_T"):
                data[name] = Double(default)
            elif vtype.upper() in ("VS", "VECTOR<SHORT>"):
                data[name] = ROOT.vector("short")()
            elif vtype.upper() in ("VUS", "VECTOR<UNSIGNED SHORT>"):
                data[name] = ROOT.vector("unsigned short")()
            elif vtype.upper() in ("VI", "VECTOR<INT>"):
                data[name] = ROOT.vector("int")()
            elif vtype.upper() in ("VUI", "VECTOR<UNSIGNED INT>"):
                data[name] = ROOT.vector("unsigned int")()
            elif vtype.upper() in ("VL", "VECTOR<LONG>"):
                data[name] = ROOT.vector("long")()
            elif vtype.upper() in ("VF", "VECTOR<FLOAT>"):
                data[name] = ROOT.vector("float")()
            elif vtype.upper() in ("VD", "VECTOR<DOUBLE>"):
                data[name] = ROOT.vector("double")()
            elif vtype.upper() in ("VVI", "VECTOR<VECTOR<INT> >"):
                data[name] = ROOT.vector("vector<int>")()
            elif vtype.upper() in ("VVUI", "VECTOR<VECTOR<UNSIGNED INT> >"):
                data[name] = ROOT.vector("vector<unsigned int>")()
            elif vtype.upper() in ("VVL", "VECTOR<VECTOR<LONG> >"):
                data[name] = ROOT.vector("vector<long>")()
            elif vtype.upper() in ("VVUL", "VECTOR<VECTOR<UNSIGNED LONG> >"):
                data[name] = ROOT.vector("vector<unsigned long>")()
            elif vtype.upper() in ("VVF", "VECTOR<VECTOR<FLOAT> >"):
                data[name] = ROOT.vector("vector<float>")()
            elif vtype.upper() in ("VVD", "VECTOR<VECTOR<DOUBLE> >"):
                data[name] = ROOT.vector("vector<double>")()
            elif vtype.upper() in ("VVSTR", "VECTOR<VECTOR<STRING> >"):
                data[name] = ROOT.vector("vector<string>")()
            elif vtype.upper() in ("VSTR", "VECTOR<STRING>"):
                data[name] = ROOT.vector("string")()
            elif vtype.upper() in ("MSI", "MAP<STRING,INT>"):
                data[name] = ROOT.map("string,int")()
            elif vtype.upper() in ("MSF", "MAP<STRING,FLOAT>"):
                data[name] = ROOT.map("string,float")()
            elif vtype.upper() in ("MSS", "MAP<STRING,STRING>"):
                data[name] = ROOT.map("string,string")()
            else:
                raise TypeError("Unsupported variable type for branch %s: %s"%(name, vtype.upper()))
            if name in methods or name.startswith("_"):
                raise ValueError("Illegal variable name: %s"%name)
        return data
    
    def reset(self):
        
        for value in self.values():
            value.clear()

    def flat(self, variables = None):

        if variables is None:
            variables = self.variables
        else:
            variables = filter(lambda a: a[0] in variables, self.variables)
        return TreeBuffer(variables, flatten = True)
    
    def update(self, variables):

        if not isinstance(variables, TreeBuffer):
            variables = self.__process(variables)
        dict.update(self, variables)
    
    def __setattr__(self, attr, value):
        """
        Maps attributes to values.
        Only if we are initialised
        """
        # this test allows attributes to be set in the __init__ method
        if not self.__dict__.has_key("_%s__initialised" % self.__class__.__name__):
            return super(TreeBuffer, self).__setattr__(attr, value)
        elif self.__dict__.has_key(attr): # any normal attributes are handled normally
            return super(TreeBuffer, self).__setattr__(attr, value)
        elif attr in self:
            variable = self.__getitem__(attr)
            if isinstance(variable, Variable):
                variable.set(value)
                return
            raise AttributeError("cannot set non-Variable type attribute '%s' of %s instance" % (attr, self.__class__.__name__))
        raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, attr))
    
    def __getattr__(self, attr):

        try:
            variable = self[attr]
            if isinstance(variable, Variable):
                return variable.value
            return variable
        except KeyError:
            raise AttributeError("%s instance has no attribute '%s'" % (self.__class__.__name__, attr))
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        rep = ""
        for var, value in self.items():
            rep += "%s ==> %s\n"%(var, value)
        return rep
