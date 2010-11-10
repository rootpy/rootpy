import TauVariables
import TruthTools
        
class Jet:
    
    def __init__(self, jet, type="RECO"):
        
        if jet == None:
            raise ValueError()
        self.jet = jet
        self.e = TauVariables.DEFAULT
        self.et = TauVariables.DEFAULT
        self.pt = TauVariables.DEFAULT
        self.eta = TauVariables.DEFAULT
        self.pdgid = TauVariables.DEFAULT
        self.numTracks = TauVariables.DEFAULT

        if type == "TAU":
            self.numTracks = TruthTools.numProng(jet)
            visSum = TruthTools.getTrueTauVisibleSum(jet)
            self.e = visSum.e()
            self.et = visSum.et()
            self.pt = visSum.perp()
            self.eta = visSum.eta()
        else:
            self.e = jet.e()
            self.et = jet.et()
            self.pt = jet.pt()
            self.eta = jet.eta()

        if type == "RECO":
            self.numTracks = jet.numTrack()

        self.match = None

    def setMatch(self,jet):
        
        self.match = jet
        
    def momentum(self):
        
        return self.jet.hlv()
    
    def __repr__(self):
        
        return self.__str__()
    
    def __str__(self):
        
        return "(eta:% .2f, phi:% .2f, ET: %.2f, E: %.2f)"%(self.eta,self.jet.phi(),self.et,self.e)
