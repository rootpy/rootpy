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
            if type.upper() == "I":
                data[name] = Int(default)
            elif type.upper() == "F":
                data[name] = Float(default)
            elif type.upper() == "VI":
                data[name] = ROOT.vector("int")()
            elif type.upper() == "VF":
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
