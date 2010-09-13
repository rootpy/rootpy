from Types import *
from UserDict import UserDict

class NtupleBuffer(UserDict):

    def __init__(self,variables,default=-1111):
        
        data = {}
        for name,type in variables:
            if type.upper() == "I":
                data[name] = Int(default)
            elif type.upper() == "F":
                data[name] = Float(default)
            else:
                raise TypeError("Unsupported variable type: %s"%(type.upper()))
        UserDict.__init__(self,data)

    def reset(self):
        
        for value in self.values():
            value.reset()
