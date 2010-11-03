import math
from ROOTPy.utils import *

def toRel16Tracking(tree):

    for itau in range(len(tree.tau_Et)):
        tau_numTrack = tree.tau_numTrack[itau]
        tau_eta = tree.tau_seedCalo_eta[itau]
        tau_phi = tree.tau_seedCalo_phi[itau]
        new_numTrack = 0
        leadTrkPt = 0.

        if tau_numTrack != tree.tau_track_n[itau]:
            print "tau_numTrack (%i) != tau_track_n (%i)"%(tau_numTrack,tree.tau_track_n[itau])

        # recount tracks with rel 16 selection
        for itrack in range(tau_numTrack):
            track_eta = tree.tau_track_eta[itau][itrack]
            track_phi = tree.tau_track_phi[itau][itrack]
            track_d0 = tree.tau_track_atPV_d0[itau][itrack]
            track_theta = tree.tau_track_atPV_theta[itau][itrack]
            track_z0 = tree.tau_track_atPV_z0[itau][itrack]
            track_nBLHits = tree.tau_track_nBLHits[itau][itrack]
            track_nPixHits = tree.tau_track_nPixHits[itau][itrack]
            track_pt = tree.tau_track_pt[itau][itrack]
            dr = dr(tau_eta, tau_phi, track_eta, track_phi)

            if ( dr < 0.2 ) and \
               ( track_nBLHits >= 1 ) and \
               ( track_nPixHits >= 2 ) and \
               ( track_d0 < 1.0 ) and \
               ( track_z0 * math.sin(track_theta) < 1.5 ):
                new_numTrack += 1
                if track_pt > leadTrkPt:
                    leadTrkPt = track_pt

        # correct numTrack
        tree.tau_numTrack[itau] = new_numTrack

        # correct leadTrkPt
        tree.tau_leadTrkPt[itau] = leadTrkPt

        # correct etOverPtLeadTrk
        tree.tau_etOverPtLeadTrk[itau] = ( tree.tau_seedCalo_etEMAtEMScale[itau] + tree.tau_seedCalo_etHadAtEMScale[itau] )/ leadTrkPt
