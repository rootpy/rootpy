from ROOTPy.analysis.filtering import Filter
from math import *
from operator import itemgetter
from ROOTPy.utils import *

class DiTauLeadSub(Filter):
"""used on data only"""
    
    def passes(self):

        # there must be at least two taus
        if self.buffer.tau_n.size()<2:
            return False
        # find leading and subleading taus
        ets = zip(self.buffer.tau_Et,range(self.buffer.tau_n)
        ets.sort(key=itemgetter(0), reverse=True) # sort descending by Et
        
        # require that the leading tau ET>30GeV and subleading tau ET>15GeV
        if ets[0][0] <= 30000 or ets[1][0] <= 15000:
            return False
        
        leading_index = ets[0][1]
        subleading_index = ets[1][1]


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

class DiTau(Filter):

    def passes(self):

        # there must be at least two taus
        if self.buffer.tau_n.size()<2:
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

class Triggers(Filter):

    def passes(self):

        if not (self.buffer.L1_J5 == 1 or self.buffer.L1_J10 == 1 or self.buffer.L1_J30 == 1 or self.buffer.L1_J55 == 1):
            return False
        return True

class JetCleaning(Filter):
"""Winter 2011 jet cleaning (used on data only)"""
    
    def passes(self):

        for itau in range(self.buffer.tau_n):
            tau_author = (self.buffer.tau_author)[i_tau];
            tau_Et = (self.buffer.tau_Et)[i_tau];
            if ( tau_Et > 15000.0 ) and ( tau_author == 1 || tau_author == 3 ):
                if ((self.buffer.tau_jet_emfrac)[i_tau]>0.95 and abs((self.buffer.tau_jet_quality)[i_tau])>0.8): return False # EM coherent noise
                if ((self.buffer.tau_jet_hecf)[i_tau]>0.8 and (self.buffer.tau_jet_n90)[i_tau]<=5): return False # HEC spike
                if ((self.buffer.tau_jet_hecf)[i_tau]>0.5 and abs((self.buffer.tau_jet_quality)[i_tau])>0.5): return False # HEC spike
                if (abs((self.buffer.tau_jet_timing)[i_tau])>25.0): return False # Cosmics, beam background
                if ((self.buffer.tau_jet_fracSamplingMax)[i_tau]>0.99 and abs((self.buffer.tau_eta)[i_tau])<2.0): return False # Cosmics, beam background
                if ((self.buffer.tau_jet_emfrac)[i_tau]<0.05 and (self.buffer.tau_numTrack)[i_tau]==0): return False # specific for tau by Koji.
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
        

