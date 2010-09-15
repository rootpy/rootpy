import ROOT
from Types import *

ROOT.gROOT.ProcessLine('.L dicts.C+')

demote = {"Float_T":"F",
          "Int_T":"I",
          "Int":"I",
          "Float":"F",
          "F":"F",
          "I":"I",
          "vector<float>":"F",
          "vector<int>":"I",
          "vector<int,allocator<int> >":"I",
          "vector<float,allocator<float> >":"F",
          "VF":"F",
          "VI":"I",
          "vector<vector<float> >":"VF",
          "vector<vector<float> >":"VI",
          "vector<vector<int>,allocator<vector<int> > >":"VI",
          "vector<vector<float>,allocator<vector<float> > >":"VF",
          "VVF":"VF",
          "VVI":"VI"}

class NtupleBuffer(dict):

    def __init__(self,variables,default=-1111,flatten=False):
        
        data = {}
        methods = dir(self)
        processed = []
        for name,type in variables:
            if flatten:
                type = demote[type]
            if name in processed:
                raise ValueError("Duplicate variable name %s"%name)
            else:
                processed.append(name)
            if type.upper() in ("I","INT_T"):
                data[name] = Int(default)
            elif type.upper() in ("F","FLOAT_T"):
                data[name] = Float(default)
            elif type.upper() in ("VI","VECTOR<INT>"):
                data[name] = ROOT.vector("int")()
            elif type.upper() in ("VF","VECTOR<FLOAT>"):
                data[name] = ROOT.vector("float")()
            elif type.upper() in ("VVI","VECTOR<VECTOR<INT> >"):
                data[name] = ROOT.vector("vector<int>")()
            elif type.upper() in ("VVF","VECTOR<VECTOR<FLOAT> >"):
                data[name] = ROOT.vector("vector<float>")()
            else:
                raise TypeError("Unsupported variable type: %s"%(type.upper()))
            if name not in methods and not name.startswith("_"):
                setattr(self,name,data[name])
            else:
                raise ValueError("Illegal variable name: %s"%name)
        dict.__init__(self,data)

    def reset(self):
        
        for value in self.values():
            value.clear()
    
    def fuse(self,tree,variables=None,createMissing=False):

        tree.ResetBranchAddresses()
        tree.SetBranchStatus("*",False)
        if variables == None:
            variables = self.keys()
        for var in variables:
            value = self[var]
            if not tree.GetBranch(var):
                if createMissing:
                    tree.Branch(var,value)
                else:
                    raise ValueError("Tree %s does not have a branch named %s"%(tree.GetName(),var))
            else:
                tree.SetBranchAddress(var,value)
            tree.SetBranchStatus(var,True)
