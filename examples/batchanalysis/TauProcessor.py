#import PyCintex
import ROOT, glob, sys, array, traceback
from math import *
import rootpy.datasets as datasets
from variables import *
from rootpy.analysis.filtering import FilterList, GRL
from taufilters import *
from taurecalcvars import *
from rootpy.analysis.batch import Student
from rootpy.ntuple import Ntuple, NtupleBuffer, NtupleChain

dicts =\
"""
#include <vector>
#ifdef __CINT__
#pragma link C++ class vector<vector<int> >;
#pragma link C++ class vector<vector<float> >;
#else
template class std::vector<std::vector<int> >;
template class std::vector<std::vector<float> >;
#endif
"""

try:
    a = ROOT.vector("vector<float>")
except:
    tmpfile = open("dicts.C",'w')
    tmpfile.write(dicts)
    tmpfile.close()
    level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    ROOT.gROOT.ProcessLine('.L dicts.C+')
    import os
    os.unlink("dicts.C")
    os.unlink("dicts_C.d")
    os.unlink("dicts_C.so")
    ROOT.gErrorIgnoreLevel = level

#ROOT.gSystem.CompileMacro( 'EMJESfix.hpp')
ROOT.gErrorIgnoreLevel = ROOT.kFatal

class TauProcessor(Student):
    
    def __init__( self, name, files, treename, datatype, classtype, weight, numEvents = -1, pipe=None, doJESsys=False, grl=None):
    
        Student.__init__( self, name, files, treename, datatype, classtype, weight, numEvents, pipe)
        self.tree = None
        self.doTruth = False
        if classtype == datasets.classes['SIGNAL']:
            self.doTruth = True
        self.doJESsys=doJESsys
        if doJESsys:
            self.jetEMJESfixer = ROOT.EMJESFixer()
        self.grl = grl

    def coursework(self):

        Student.coursework(self)
        
        variablesIn = [
            ('tau_charge','VI'),
            ("tau_eta","VF"),
            ("tau_phi","VF"),
            ("tau_Et","VF"),
            ("tau_etOverPtLeadTrk","VF"),
            ("tau_calcVars_topoInvMass","VF"),
            ("tau_calcVars_emFracCalib","VF"),
            ('tau_calcVars_sumTrkPt','VF'),
            ("tau_massTrkSys","VF"),
            ("tau_trkWidth2","VF"),
            ("tau_author","VI"),
            ("tau_BDTJetScore","VF"),
            ("tau_likelihood","VF"),
            ("tau_ipZ0SinThetaSigLeadTrk","VF"),
            ("tau_leadTrkPt","VF"),
            ("tau_ipSigLeadTrk","VF"),
            ("tau_ipSigLeadLooseTrk","VF"),
            ("tau_leadLooseTrkPt","VF"),
            ("tau_chrgLooseTrk","VF"),
            ("tau_trFlightPathSig","VF"),
            ("tau_etEflow","VF"),
            ("tau_mEflow","VF"),
            ("tau_seedCalo_etEMAtEMScale","VF"),
            ("tau_seedCalo_etHadAtEMScale","VF"),
            ("tau_seedCalo_etEMCalib","VF"),
            ("tau_seedCalo_etHadCalib","VF"),
            ("tau_seedCalo_trkAvgDist","VF"),
            ("tau_seedCalo_hadRadius","VF"),
            ("tau_seedCalo_centFrac","VF"),
            ("tau_seedCalo_EMRadius","VF"),
            ("tau_seedCalo_isolFrac","VF"),
            ("tau_seedCalo_eta","VF"),
            ("tau_seedCalo_phi","VF"),
            ("tau_seedTrk_EMRadius","VF"),
            ("tau_calcVars_effTopoMeanDeltaR","VF"),
            ("tau_calcVars_effTopoInvMass","VF"),
            ("tau_calcVars_numEffTopoClusters","VF"),
            ("tau_calcVars_topoMeanDeltaR","VF"),
            ("tau_numTrack","VI"),
            ("tau_seedCalo_nIsolLooseTrk","VI"),
            ("tau_calcVars_numTopoClusters","VI"),
            ("tau_seedTrk_nIsolTrk","VI"),
            ("tau_seedCalo_nStrip","VI"),
            ("tau_nPi0","VI"),
            ("tau_nProngLoose","VI"),
            ("tau_nLooseTrk","VI"),
            ("tau_nLooseConvTrk","VI"),
            ("tau_track_n","VI"),
            ("tau_n","I"),
            ("vxp_n","I"),
            ("RunNumber","I"),
            ("EventNumber","I")
        ]
        if self.doJESsys:
            variablesIn += [
                ("tau_cell_n","VI"),
                ("tau_jet_pt","VF"),
                ("tau_jet_eta","VF"),
                ("tau_jet_EMJES","VF")
            ]
        extraVariablesIn = [
            ("lbn","I"),
            ("L1_J5","UI"),
            ("L1_J10","UI"),
            ("L1_J30","UI"),
            ("L1_J55","UI"),
            ("L1_J75","UI"),
            ("L1_J95","UI"),
            ("L1_TAU5","UI"),
            ("jet_isGood","VI"),
            ("vxp_nTracks","VI"),
            ("tau_jet_phi","VF"),
            ('tau_jet_eta','VF'),
            ('tau_track_pt','VVF'),
            ('tau_track_eta','VVF'),
            ('tau_track_phi','VVF'),
            ('tau_track_atPV_d0','VVF'),
            ('tau_track_atPV_z0','VVF'),
            ('tau_track_atPV_theta','VVF'),
            ('tau_track_nPixHits','VVI'),
            ('tau_track_nSCTHits','VVI'),
            ('tau_track_nTRTHits','VVI'),
            ('tau_track_nBLHits','VVI')
            #('tau_track_charge','VVI') Use tau_track_qoverp
        ]
        if self.datatype == datasets.types['DATA']:
            extraVariablesIn += [
                ('trig_L1_jet_n','I'),
                ('trig_L1_jet_eta','VF'),
                ('trig_L1_jet_phi','VF'),
                ('trig_L1_jet_thrPattern','VI'),
                ('trig_L1_jet_thrValues','VVF'),
                ('tau_jet_emfrac','VF'),
                ('tau_jet_quality','VF'),
                ('tau_jet_hecf','VF'),
                ('tau_jet_n90','VF'),
                ('tau_jet_timing','VF'),
                ('tau_jet_fracSamplingMax','VF')
            ]
        
        variablesOut = [
            ("tau_calcVars_emFracEMScale","VF")
        ]

        if self.doJESsys:
            extraVariablesIn += [
                ("tau_cluster_E","VVF"),
                ("tau_cluster_eta","VVF"),
                ("tau_cluster_phi","VVF")
            ]
            variablesOut += [
                ("tau_Et_EMJES","VF"),
                ("tau_etOverPtLeadTrk_EMJES","VF"),
                ("tau_calcVars_topoInvMass_EMJES","VF"),
                ("tau_calcVars_topoInvMass_recalc","VF"),
                ("tau_calcVars_emFracCalib_EMJES","VF")
            ]
       
        truthVariables = [] 
        extraTruthVariablesIn = []
        extraTruthVariablesOut = []
        if self.doTruth:
            extraVariablesIn += [
                ("trueTau_tauAssocSmall_index","VI"),
                ("tau_trueTauAssocSmall_index","VI")
            ]
            variablesOut += [
                ("tau_nProngOfMatch","VI"),
                ("tau_isTruthMatched","VI"),
                ("tau_EtOfMatch","VF"),
                ("tau_etaOfMatch","VF")
            ]

            extraTruthVariablesIn += [
                ("trueTau_nProng", "VI" ),
                ("trueTau_vis_Et", "VF" ),
                ("trueTau_vis_eta", "VF" )
            ]
            extraTruthVariablesOut += [
                ('EventNumber','I'),
                ('vxp_n','I'),
                ('tau_eta','VF'),
                ('tau_Et','VF'),
                ('tau_isTruthMatched','VI'),
                ('tau_EtOfMatch','VF'),
                ("tau_etaOfMatch","VF"),
                ('tau_nProng','VI')
            ]
                        
        self.variables = [ var for var,type in variablesIn ]
        self.variablesOutExtra = [ var for var,type in variablesOut ]

        self.buffer = NtupleBuffer(variablesIn+extraVariablesIn+truthVariables+extraTruthVariablesIn)
        self.tree = NtupleChain(self.treename,files=self.files,buffer=self.buffer)
        #self.buffer.fuse(self.tree)
        #self.tree.SetBranchAddress("tau_Et",self.buffer.tau_Et)
        self.bufferOut = NtupleBuffer(variablesIn+variablesOut,flatten=True)
        self.output.cd()
        self.D4PD = Ntuple(self.name,buffer=self.bufferOut)
        #self.D4PD.SetWeight(self.weight)
        if self.doTruth:
            self.bufferOutTruth = NtupleBuffer(truthVariables+extraTruthVariablesOut,flatten=True)
            self.D4PDTruth = Ntuple("%s_truth"%self.name,buffer=self.bufferOutTruth)
            #self.D4PDTruth.SetWeight(self.weight)
        if self.datatype == datasets.types['DATA']:
            if self.grl != None:
                self.filters = FilterList([GRL(self.tree,self.grl),Triggers(self.tree),PriVertex(self.tree),JetCleaning(self.tree),LeadTauTrigMatch(self.tree),DiTau(self.tree)])
            else:
                self.filters = FilterList([Triggers(self.tree),PriVertex(self.tree),JetCleaning(self.tree),LeadTauTrigMatch(self.tree),DiTau(self.tree)])
        else:
            self.filters = FilterList([Triggers(self.tree),PriVertex(self.tree)])

    #__________________________________________________________________
    def research(self):

        Student.research(self)
        
        if self.event == self.numEvents:
            return False
        if not self.tree.read():
            return False
        self.event += 1

        if self.filters:
            
            leadTau = -1
            if self.datatype == datasets.types['DATA']:
                # find index of lead tau
                highET = 0.
                for itau,et in enumerate(self.tree.tau_Et):
                    if et > highET:
                        highET = et
                        leadTau = itau
            
            # recalculate variables
            toRel16Tracking(self.tree)
            
            # loop over taus to fill ntuple 
            for itau in xrange(self.tree.tau_n[0]):
                 
                # exclude the leading tau in data
                if self.datatype == datasets.types['DATA'] and itau == leadTau:
                    continue

                # only keep taus above 15GeV
                if self.tree.tau_Et[itau]<=15000.:
                    continue
                
                for var in self.variables:
                    self.bufferOut[var].set(self.buffer[var][itau])

                totET = self.tree.tau_seedCalo_etEMAtEMScale[itau] + self.tree.tau_seedCalo_etHadAtEMScale[itau]
                if totET != 0:
                    self.bufferOut['tau_calcVars_emFracEMScale'][0] = self.tree.tau_seedCalo_etEMAtEMScale[itau] / totET
                else:
                    self.bufferOut['tau_calcVars_emFracEMScale'][0] = -1111.
                
                if self.doJESsys:
                    # Energy scale recalculation:

                    # EM scale 
                    tau_Et_EM = self.tree.tau_seedCalo_etEMAtEMScale[itau] + self.tree.tau_seedCalo_etHadAtEMScale[itau]
                    
                    # EM+JES scale
                    tau_GCWScale = self.tree.tau_seedCalo_etEMCalib[itau] + self.tree.tau_seedCalo_etHadCalib[itau]
                    if tau_GCWScale > 0:
                        tau_GCWandFF = self.tree.tau_Et[itau]
                        tau_EMJES = self.tree.tau_jet_EMJES[itau]
                        if tau_EMJES == 0:
                            tau_EMJES = self.jetEMJESfixer.fixAntiKt4H1Topo(self.tree.tau_jet_pt[itau],self.tree.tau_jet_eta[itau])
                        tau_EMJES_FF = tau_EMJES*tau_GCWandFF/tau_GCWScale
                        tau_Et_EMJES = tau_Et_EM*tau_EMJES_FF
                        self.bufferOut['tau_Et_EMJES'][0] = tau_Et_EMJES


                        if self.tree.tau_leadTrkPt[itau] > 0:
                            self.bufferOut['tau_etOverPtLeadTrk_EMJES'][0] = tau_Et_EMJES / self.tree.tau_leadTrkPt[itau]
                        else:
                            self.bufferOut['tau_etOverPtLeadTrk_EMJES'][0] = self.bufferOut['tau_etOverPtLeadTrk'][0]
                        
                        self.bufferOut['tau_calcVars_emFracCalib_EMJES'][0] = self.tree.tau_seedCalo_etEMAtEMScale[itau] * tau_EMJES_FF / tau_Et_EMJES
                        
                        clusters_EMJES = getClusters(energies=self.tree.tau_cluster_E[itau],
                                               etas=self.tree.tau_cluster_eta[itau],
                                               phis=self.tree.tau_cluster_phi[itau],
                                               energyScale=tau_EMJES*tau_GCWandFF/tau_GCWScale)
                        
                        clusters = getClusters(energies=self.tree.tau_cluster_E[itau],
                                               etas=self.tree.tau_cluster_eta[itau],
                                               phis=self.tree.tau_cluster_phi[itau])

                        topoMass_EMJES = topoClusterMass(clusters_EMJES)
                        topoMass = topoClusterMass(clusters)
                        
                        self.bufferOut['tau_calcVars_topoInvMass_EMJES'][0] = topoMass_EMJES
                        self.bufferOut['tau_calcVars_topoInvMass_recalc'][0] = topoMass
                    else:
                        self.bufferOut['tau_Et_EMJES'][0] = self.bufferOut['tau_Et'][0]
                        self.bufferOut['tau_etOverPtLeadTrk_EMJES'][0] = self.bufferOut['tau_etOverPtLeadTrk'][0]
                        self.bufferOut['tau_calcVars_emFracCalib_EMJES'][0] = self.bufferOut['tau_calcVars_emFracCalib'][0]
                        self.bufferOut['tau_calcVars_topoInvMass_EMJES'][0] = self.bufferOut['tau_calcVars_topoInvMass'][0]
                        self.bufferOut['tau_calcVars_topoInvMass_recalc'][0] = self.bufferOut['tau_calcVars_topoInvMass'][0]
                        
                # truth variables to be calculated per reco tau
                if self.doTruth:
                    if self.tree.tau_trueTauAssocSmall_index[itau] >= 0:
                        self.bufferOut['tau_nProngOfMatch'].set(self.tree.trueTau_nProng[self.tree.tau_trueTauAssocSmall_index[itau]])
                        self.bufferOut['tau_EtOfMatch'].set(self.tree.trueTau_vis_Et[self.tree.tau_trueTauAssocSmall_index[itau]])
                        self.bufferOut['tau_etaOfMatch'].set(self.tree.trueTau_vis_eta[self.tree.tau_trueTauAssocSmall_index[itau]])
                        self.bufferOut['tau_isTruthMatched'].set(1)
                    else:
                        self.bufferOut['tau_nProngOfMatch'].set(-1111)
                        self.bufferOut['tau_EtOfMatch'].set(-1111.)
                        self.bufferOut['tau_etaOfMatch'].set(-1111.)
                        self.bufferOut['tau_isTruthMatched'].set(0)
                # fill ntuple once per tau
                self.D4PD.Fill()
            # Now loop over true taus and fill ntuple once per truth tau
            if self.doTruth:
                self.bufferOutTruth['EventNumber'].set(self.tree.EventNumber)
                self.bufferOutTruth['vxp_n'].set(self.tree.vxp_n)
                for itrue in xrange( self.tree.trueTau_vis_Et.size() ): 
                    self.bufferOutTruth['tau_nProng'].set(self.tree.trueTau_nProng[itrue])
                    self.bufferOutTruth['tau_Et'].set(self.tree.trueTau_vis_Et[itrue])
                    self.bufferOutTruth['tau_eta'].set(self.tree.trueTau_vis_eta[itrue])
                    matchIndex = self.tree.trueTau_tauAssocSmall_index[itrue]
                    if matchIndex >= 0:
                        self.bufferOutTruth['tau_EtOfMatch'].set(self.tree.tau_Et[matchIndex])
                        self.bufferOutTruth['tau_etaOfMatch'].set(self.tree.tau_eta[matchIndex])
                        self.bufferOutTruth['tau_isTruthMatched'].set(1)
                    else:
                        self.bufferOutTruth['tau_EtOfMatch'].set(-1111.)
                        self.bufferOutTruth['tau_etaOfMatch'].set(-1111.)
                        self.bufferOutTruth['tau_isTruthMatched'].set(0)
                    self.D4PDTruth.Fill()
        return True
    
    def defend(self):

        Student.defend(self)

        for filter in self.filters:
            print filter

