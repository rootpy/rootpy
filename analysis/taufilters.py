from filtering import Filter
from math import *

class DiTau(Filter):

    def passes(self,buffer):

        Filter.passes(self)
        
        # diTau cut, each tau above 15GeV with delta phi
        if buffer.tau_Et.size()<2:
            return False
        # if there are 2 taus, check dPhiand Et
        goodPhi=[]
        for et,phi in zip(buffer.tau_Et,buffer.tau_phi):
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
        self.passing += 1
        return True

class L1_TAU5(Filter):

    def passes(self,buffer):

        Filter.passes(self)

        if buffer.L1_TAU5 != 1:
            return False
        self.passing += 1
        return True

class IsGood(Filter):
    
    def passes(self,buffer):

        Filter.passes(self)

        # keep only events with all good jets
        for status in buffer.jet_isGood:
            if status != 2:
                return False
        self.passing += 1
        return True

class LeadTau(Filter):

    def passes(self,buffer):

        Filter.passes(self)

        # Leading tau above 30GeV
        for et in buffer.tau_Et:
            if et>30000.:
                self.passing += 1
                return True
        return False

class PriVertex(Filter):

    def passes(self,buffer):

        Filter.passes(self)

        # vertex cut, at least one vertex with at least 4 tracks
        for vxp in buffer.vxp_nTracks:
            if vxp >= 4:
                self.passing += 1
                return True
        return False
        
