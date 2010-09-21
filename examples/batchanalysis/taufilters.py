from PyROOT.analysis.filtering import Filter
from math import *

class DiTau(Filter):

    def passes(self):

        # diTau cut, each tau above 15GeV with delta phi
        if self.buffer.tau_Et.size()<2:
            return False
        # if there are 2 taus, check dPhiand Et
        goodPhi=[]
        for et,phi in zip(self.buffer.tau_Et,self.buffer.tau_phi):
            if et>15000.:
                goodPhi.append(phi)
        if len(goodPhi) < 2:
            return False
        phiPass = False
        for phi1 in goodPhi: 
            if phiPass:       # as soon as 1 pair passes, break
                break
            for phi2 in goodPhi: # always have 1 pair with dPhi=0 ;-) 
                dPhi = phi1-phi2
                deltaPhi = fmod( dPhi+3*pi,2*pi )-pi                 
                if abs(deltaPhi)>2.7:
                    phiPass=True
                    break
        if not phiPass:
            return False
        return True

class L1_TAU5(Filter):

    def passes(self):

        if self.buffer.L1_TAU5 != 1:
            return False
        return True

class IsGood(Filter):
    
    def passes(self):

        # keep only events with all good jets
        for status in self.buffer.jet_isGood:
            if status != 2:
                return False
        return True

class LeadTau(Filter):

    def passes(self):

        # Leading tau above 30GeV
        for et in self.buffer.tau_Et:
            if et>30000.:
                return True
        return False

class PriVertex(Filter):

    def passes(self):

        # vertex cut, at least one vertex with at least 4 tracks
        for vxp in self.buffer.vxp_nTracks:
            if vxp >= 4:
                return True
        return False
        
