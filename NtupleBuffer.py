import ROOT
from Types import *

ROOT.gROOT.ProcessLine('.L dicts.C+')

class NtupleBuffer(dict):

    def __init__(self,variables,default=-1111):
        
        data = {}
        methods = dir(self)
        processed = []
        for name,type in variables:
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
    
    def fuse(self,tree):

        tree.ResetBranchAddresses()
        for var,value in self.items():
            if not tree.GetBranch(var):
                tree.Branch(var,value)
            else:
                tree.SetBranchAddress(var,value)
