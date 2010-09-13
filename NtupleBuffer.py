from Types import *
from UserDict import UserDict

class NtupleBuffer(UserDict):

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
            else:
                raise TypeError("Unsupported variable type: %s"%(type.upper()))
            if name not in methods:
                setattr(self,name,data[name])
            else:
                raise ValueError("Illegal variable name conflicts with class method %s"%name)
        UserDict.__init__(self,data)

    def reset(self):
        
        for value in self.values():
            value.reset()
