import time
import re
import ROOT
from rootpy.basictypes import *
from rootpy.classfactory import *
from rootpy.core import *
from rootpy.utils import *
from rootpy.registry import *
from rootpy.io import *
from rootpy.filtering import *
from rootpy.plotting import *

class VarProxy(object):

    def __init__(self, tree, prefix, index, collections=None):

        self.index = index
        self.tree = tree
        self.prefix = prefix
        self.collections = collections

    def __getitem__(self, thing):

        return getattr(self, thing)
         
    def __getattr__(self, attr):

        if attr.startswith(self.prefix):
            return getattr(self.tree, attr)[self.index]
        return getattr(self.tree, self.prefix + attr)[self.index]

class TreeCollection(object):

    def __init__(self, tree, prefix, size):
        
        self.tree = tree
        self.prefix = prefix
        self.size = size
        self.subcollections = []
        super(TreeCollection, self).__init__()

    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError()
        return VarProxy(self.tree, self.prefix, index, self.subcollections)

    def __len__(self):

        return getattr(self.tree, self.size).value()
    
    def collection(self, name, prefix, size):
        
        self.subcollections.append(TreeCollection(self.tree, self.prefix + prefix, size, self.subcollections))

    def __iter__(self):

        for index in xrange(len(self)):
            yield VarProxy(self.tree, self.prefix, index)


class Tree(Plottable, Object, ROOT.TTree):
    """
    Inherits from TTree so all regular TTree methods are available
    but Draw has been overridden to improve usage in Python
    """
    draw_command = re.compile('^.+>>[\+]?(?P<name>[^(]+).*$')

    def __init__(self, name = None, title = None):

        Object.__init__(self, name, title)
    
    def _post_init(self):

        Plottable.__init__(self)
        self.buffer = None
    
    def set_buffer(self, buffer):

        self.buffer = buffer
        for attr, value in buffer.items():
            setattr(self, attr, value)

    def set_branches_from_buffer(self, buffer, variables = None):
    
        for name, value in buffer.items():
            if variables is not None:
                if name not in variables:
                    continue
            if isinstance(value, Variable):
                self.Branch(name, value, "%s/%s"% (name, value.type()))
            else:
                self.Branch(name, value)
        self.set_buffer(buffer)

    def set_addresses_from_buffer(self, buffer, variables = None):
        
        for name, value in buffer.items():
            if variables is not None:
                if name not in variables:
                    continue
            if self.GetBranch(name):
                self.SetBranchAddress(name, value)
        self.set_buffer(buffer)

    def get_buffer(self):

        if self.buffer is not None:
            return self.buffer
        buffer = []
        for branch in self.iterbranches():
            if self.GetBranchStatus(branch.GetName()):
                typename = branch.GetClassName()
                if not typename:
                    typename = branch.GetListOfLeaves()[0].GetTypeName()
                buffer.append((branch.GetName(), typename))
        return TreeBuffer(buffer)
    
    def activate(self, variables, exclusive=True):

        if exclusive:
            self.SetBranchStatus('*',0)
        for variable in variables:
            if self.GetBranch(variable):
                self.SetBranchStatus(variable,1)

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

    def Draw(self, *args):
        """
        Draw a TTree with a selection as usual, but return the created histogram.
        """ 
        if len(args) == 0:
            raise TypeError("Draw did not receive any arguments")
        match = re.match(Tree.draw_command, args[0])
        histname = None
        if match:
            histname = match.group('name')
            hist_exists = ROOT.gDirectory.Get(histname) is not None
        ROOT.TTree.Draw(self, *args)
        if histname is not None:
            hist = asrootpy(ROOT.gDirectory.Get(histname))
            # if the hist already existed then I will
            # not overwrite its plottable features
            if not hist_exists and isinstance(hist, Plottable):
                hist.decorate(self)
            return hist
        else:
            hist = asrootpy(ROOT.gPad.GetPrimitive("htemp"))
            if isinstance(hist, Plottable):
                hist.decorate(self)
            return hist

register(Tree, Tree._post_init)

class TreeChain:
    """
    A replacement for TChain
    """ 
    def __init__(self, name, files, buffer=None, branches=None, events=-1):
        
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
        if self.buffer:
            for attr, value in self.userbuffer.items():
                setattr(self, attr, value)
        self.weight = 1.
        self.tree = None
        self.file = None
        self.filters = EventFilterList()
        self.userdata = {}
        self.file_change_hooks = []
        self.events = events
        
    def init(self):
        
        if not self.__initialize():
            raise RuntimeError("unable to initialize TreeChain")
    
    def collection(self, name, prefix, size):
        
        coll = TreeCollection(self, prefix, size)
        setattr(self, name, coll)
    
    def add_file_change_hook(self, target, args):
    
        self.file_change_hooks.append((target, args))

    def __initialize(self):

        if self.tree != None:
            self.tree = None
        if self.file != None:
            self.file.Close()
            self.file = None
        if len(self.files) > 0:
            print "%i files remaining to process"% len(self.files)
            fileName = self.files.pop()
            self.file = File(fileName)
            if not self.file:
                print "WARNING: Skipping file. Could not open file %s"%(fileName)
                return self.__initialize()
            self.tree = self.file.Get(self.name)
            if not self.tree:
                print "WARNING: Skipping file. Tree %s does not exist in file %s"%(self.name, fileName)
                return self.__initialize()
            if len(self.tree.GetListOfBranches()) == 0:
                # Try the next file:
                print "WARNING: skipping tree with no branches in file %s"%fileName
                return self.__initialize()
            if self.branches is not None:
                self.tree.activate(self.branches)
            if self.buffer is None:
                buffer = self.tree.get_buffer()
                self.buffer = buffer
                for attr, value in buffer.items():
                    setattr(self, attr, value)
            self.tree.set_addresses_from_buffer(self.buffer)
            self.weight = self.tree.GetWeight()
            for target, args in self.file_change_hooks:
                target(*args, name=self.name, file=self.file)
            return True
        return False
    
    def set_filters(self, filterlist):
        
        self.filters = filterlist

    def append_filter(self, filter):

        self.filters.append(filter)

    def prepend_filter(self, filter):

        self.filters.insert(0, filter)
    
    def __getitem__(self, item):

        return self.tree.__getitem__(item)

    def __iter__(self):
        
        events = 0
        while True:
            t1 = time.time()
            entries = 0
            for entry in self.tree:
                entries += 1
                self.userdata = {}
                if self.filters(self):
                    yield self
                    events += 1
                    if self.events == events:
                        break
            if self.events == events:
                break
            print "%i entries per second"% int(entries / (time.time() - t1))
            if not self.__initialize():
                break

class TreeBuffer(dict):
    """
    A dictionary mapping variable names to values
    """
    generate("vector<vector<float> >", "<vector>")
    generate("vector<vector<int> >", "<vector>")

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
              "vector<vector<float> >":"VI",
              "vector<vector<int>, allocator<vector<int> > >":"VI",
              "vector<vector<float>, allocator<vector<float> > >":"VF",
              "VVF":"VF",
              "VVI":"VI"} 

    def __init__(self, variables, default = -1111, flatten = False):
        
        self.variables = variables
        dict.__init__(self, self.__process(variables, default, flatten))

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
            elif vtype.upper() in ("I", "INT_T"):
                data[name] = Int(default)
            elif vtype.upper() in ("UI", "UINT_T"):
                data[name] = UInt(default)
            elif vtype.upper() in ("F", "FLOAT_T"):
                data[name] = Float(default)
            elif vtype.upper() in ("D", "DOUBLE_T"):
                data[name] = Double(default)
            elif vtype.upper() in ("VI", "VECTOR<INT>"):
                data[name] = ROOT.vector("int")()
            elif vtype.upper() in ("VUI", "VECTOR<UNSIGNED INT>"):
                data[name] = ROOT.vector("unsigned int")()
            elif vtype.upper() in ("VF", "VECTOR<FLOAT>"):
                data[name] = ROOT.vector("float")()
            elif vtype.upper() in ("VD", "VECTOR<DOUBLE>"):
                data[name] = ROOT.vector("double")()
            elif vtype.upper() in ("VVI", "VECTOR<VECTOR<INT> >"):
                data[name] = ROOT.vector("vector<int>")()
            elif vtype.upper() in ("VVF", "VECTOR<VECTOR<FLOAT> >"):
                data[name] = ROOT.vector("vector<float>")()
            else:
                raise TypeError("Unsupported variable vtype: %s"%(vtype.upper()))
            if name not in methods and not name.startswith("_"):
                setattr(self, name, data[name])
            else:
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

        data = self.__process(variables)
        dict.update(self, data)

    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        rep = ""
        for var, value in self.items():
            rep += "%s ==> %s\n"%(var, value)
        return rep
