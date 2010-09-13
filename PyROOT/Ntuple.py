import ROOT

class Ntuple(ROOT.TTree):

    def __init__(self, name, buffer):

        ROOT.TTree.__init__(self,self.name,self.name)
        for name,value in buffer.items():
            self.Branch(name, value.address(),"%s/%s"%(name,value.type()))
