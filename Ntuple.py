import ROOT
from Types import *

class Ntuple(ROOT.TTree):

    def __init__(self, name, buffer=None, variables=None):

        ROOT.TTree.__init__(self,name,name)
        if buffer != None:
            if variables == None:
                variables = buffer.keys()
            for variable in variables:
                value = buffer[variable]
                if isinstance(value,Variable):
                    self.Branch(variable, value, "%s/%s"%(name,value.type()))
                else: # Must be a ROOT.vector
                    self.Branch(variable, value)
