from Filter import *
from math import *

class DiTau(Filter):

    def passes(self,buffer):

        Filter.passes(self)
        
        # diTau cut, each tau above 15GeV with delta phi
        phiPass = False
        if buffer.tau_Et.size()<2:
            return False
        
        # if there are 2 taus, check dPhiand Et
        goodPhi=[]
        for itau in xrange(buffer.tau_Et.size()):
            if buffer.tau_Et[itau]>15000.:
                goodPhi.append(buffer.tau_phi[itau])
        if len(goodPhi) < 2:
            return False
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

class IsGood(Filter):
    
    def passes(self,buffer):

        Filter.passes(self)

        # keep only events with all good jets
        for ijet in xrange(buffer.jet_isGood.size()): # loop over jets
            if not buffer.jet_isGood[ijet]==2:
                return False
        self.passing += 1
        return True

class LeadTau(Filter):

    def passes(self,buffer):

        Filter.passes(self)

        # Leading tau above 30GeV
        passLeadTau=False
        for itau in xrange(buffer.tau_Et.size()):
            if buffer.tau_Et[itau]>30000.:
                passLeadTau = True
        if not passLeadTau:
            return False
        self.passing += 1
        return True

class PriVertex(Filter):

    def passes(self,buffer):

        Filter.passes(self)

        # vertex cut, at least one vertex with at least 4 tracks
        primaryVtxCount = 0
        for ivtx in xrange(buffer.vxp_nTracks.size()):
            if buffer.vxp_nTracks[ivtx]>=4:            
                primaryVtxCount = primaryVtxCount + 1            
        if primaryVtxCount<1:
            return False

        self.passing += 1
        return True
        
