import ROOT
import PyCintex

class pyTauDetailsManager:
    
    def __init__(self):
        
        self.manager = ROOT.TauDetailsManager()
        
    def update(self, tauJet):
        
        return self.manager.update(tauJet)
    
    def getFloatDetailValue(self,name):
        
        return self.manager.getFloatDetailValue(name)
    
    def getIntDetailValue(self,name):
        
        return self.manager.getIntDetailValue(name)