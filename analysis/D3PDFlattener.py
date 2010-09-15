import ROOT, glob, sys, array, traceback
from math import *
import datasets
from variables import *
from FilterList import *
from UserFilters import *

ROOT.gErrorIgnoreLevel = ROOT.kFatal

class D3PDFlattener( ROOT.TPySelector ):
    
    def __init__( self ):
    
        ROOT.TPySelector.__init__( self )
        self.branches={}
        self.eventfilters = FilterList([IsGood(),PriVertex(),LeadTau(),DiTau()])
        self.doTruth = False
        self.jetEMJESfixer = ROOT.EMJESFixer()

    #__________________________________________________________________
    def Begin( self ):
        
        print 'py: in Begin()'
    
    #__________________________________________________________________
    def SlaveBegin(self, tree = None):
        
        print 'py: in SlaveBegin()'
        self.LoadHistograms()
        self.NumProcessed = 0
    
    #__________________________________________________________________
    def Init(self, tree):
        
        print 'py: in Init()'
        self.tree = tree
        self.tree.SetBranchStatus( '*', False )
        self.InitializeEventStore( )
        self.NumProcessed = 0
        return True

    #__________________________________________________________________
    def Process(self, entry):
       
       try: 
            #print 'py: in Process()'
            if self.NumProcessed == 0:
                self.tree.Show()

            self.NumProcessed += 1
            nb = self.tree.GetEntry( entry )
            # fill the event weight variable
            self.LoadMetadata() 
            if self.eventfilters.passes(self.tree):
                # loop over taus to fill ntuple 
                for itau in xrange( self.tree.tau_Et.size() ):   # loop over taus
                    # loop over float variables and Ints separately 
                    # (tauIDApp.py insists that ints be ints)
                    # outputTreeList protects against non-existent variables
                    for tauVar in self.tauD4PDVars:
                        if tauVar in self.outputTreeList:
                            self.branches[tauVar][0] = float(eval('self.tree.'+tauVar+'[itau]'))
                    for tauVar in self.tauD4PDVarsInt:
                        if tauVar in self.outputTreeList:
                            self.branches[tauVar][0] = int(eval('self.tree.'+tauVar+'[itau]'))
                    # some non-tau variables to be filled per tau
                    for nonTauVar in self.nonTauVarsPerTauInt:
                        if nonTauVar in self.outputTreeList:
                            self.branches[nonTauVar][0] = eval('self.tree.'+nonTauVar)
                    # fill some calculated variables. Put anything you like here.
                    # Just make sure that any variable used in the calculation
                    # is also in one of your variable lists, or set the branch status to 
                    # True by hand in the initializeEventStore method.
                    # tau_n is only tau variable which is not a vector, treat separately
                    self.branches['tau_n'][0] = self.tree.tau_Et.size()
                    self.branches['tau_intAuthor'][0] = int(self.tree.tau_author[itau])   
                    self.branches['weight'][0] = self.currentSampleWeight
                    
                    # Energy scale recalculation:

                    # EM scale 
                    tau_Et_EM = self.tree.tau_seedCalo_etEMAtEMScale[itau] + self.tree.tau_seedCalo_etHadAtEMScale[itau]
                    
                    # EM+JES scale
                    tau_GCWScale = self.tree.tau_seedCalo_etEMCalib[itau] + self.tree.tau_seedCalo_etHadCalib[itau]
                    tau_GCWandFF = self.tree.tau_Et[itau]
                    tau_EMJES = self.tree.tau_jet_EMJES[itau]
                    if tau_EMJES == 0:
                        tau_EMJES = self.jetEMJESfixer.fixAntiKt4H1Topo(self.tree.tau_jet_pt[itau],self.tree.tau_jet_eta[itau])
                    tau_Et_EMJES = tau_Et_EM*tau_EMJES*tau_GCWandFF/tau_GCWScale
                    self.branches['tau_Et_EMJES'][0] = tau_Et_EMJES


                    if self.tree.tau_leadTrkPt[itau] > 0:
                        self.branches['tau_etOverPtLeadTrk_EMJES'][0] = tau_Et_EMJES / self.tree.tau_leadTrkPt[itau]
                    else:
                        self.branches['tau_etOverPtLeadTrk_EMJES'][0] = -1111.
                    
                    self.branches['tau_calcVars_emFracCalib_EMJES'][0] = self.tree.tau_seedCalo_etEMAtEMScale[itau] * (tau_EMJES*tau_GCWandFF/tau_GCWScale) / tau_Et_EMJES
                    
                    clusters = getClusters(energies=self.tree.tau_cluster_E[itau],
                                           etas=self.tree.tau_cluster_eta[itau],
                                           phis=self.tree.tau_cluster_phi[itau],
                                           energyScale=tau_EMJES*tau_GCWandFF/tau_GCWScale)

                    topoMass_EMJES = topoClusterMass(clusters)
                    
                    self.branches['tau_calcVars_topoInvMass_EMJES'][0] = topoMass_EMJES
                    
                    # truth variables to be calculated per reco tau
                    if self.doTruth:
                        if self.tree.tau_trueTauAssocSmall_index[itau] >= 0:
                            self.branches['tau_numProngsOfMatch'][0] = self.tree.trueTau_nProng[self.tree.tau_trueTauAssocSmall_index[itau]]
                            self.branches['tau_EtVisOfMatch'][0] = self.tree.trueTau_vis_Et[self.tree.tau_trueTauAssocSmall_index[itau]]
                            self.branches['tau_EtaVisOfMatch'][0] = self.tree.trueTau_vis_eta[self.tree.tau_trueTauAssocSmall_index[itau]]
                            self.branches['tau_isTruthMatched'][0] = 1
                        else:
                            self.branches['tau_numProngsOfMatch'][0]=-1111
                            self.branches['tau_EtVisOfMatch'][0]=-1111.
                            self.branches['tau_EtaVisOfMatch'][0]=-1111.
                            self.branches['tau_isTruthMatched'][0] = 0
                    # fill ntuple once per tau
                    self.D4PD.Fill()
                # Now loop over true taus and fill ntuple once per truth tau
                if self.doTruth:
                    for itrue in xrange( self.tree.trueTau_vis_Et.size() ): 

                        self.branches['trueTau_nProng'][0] = self.tree.trueTau_nProng[itrue]
                        self.branches['trueTau_vis_Et'][0] = self.tree.trueTau_vis_Et[itrue]
                        self.branches['trueTau_vis_eta'][0] = self.tree.trueTau_vis_eta[itrue]
                        if self.tree.trueTau_tauAssocSmall_index[itrue] >= 0:
                            self.branches['trueTau_etOfMatch'][0]=self.tree.tau_Et[self.tree.trueTau_tauAssocSmall_index[itrue]]
                        else:
                            self.branches['trueTau_etOfMatch'][0]=-1111.
                        self.D4PDTruth.Fill()
       except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            sys.exit(1)
       return True

    #__________________________________________________________________
    def SlaveTerminate(self):
        
        print 'py: keeping histograms: '
        for item in self.GetOutputList():
            print '  +',item.GetName()
     
    #__________________________________________________________________        
    def Terminate(self):
        
        print 'py: in Terminate()'
        
        ofile = ROOT.TFile.Open( 'output.root', 'RECREATE' )
        for item in self.GetOutputList():
            item.Write( )
        ofile.Close()

    #__________________________________________________________________
    # These are the variables from the D3PD that you would like to keep
    def LoadVariables(self):
        
        print "Begin Load Variables"

        self.tauD4PDVars=["tau_jet_pt"]
        self.tauD4PDVars+=["tau_jet_eta"]
        self.tauD4PDVars+=["tau_jet_EMJES"]
        self.tauD4PDVars+=["tau_eta"]
        self.tauD4PDVars+=["tau_phi"]

        self.tauD4PDVars+=["tau_Et"]
        self.tauD4PDVars+=["tau_Et_EMJES"]
        
        self.tauD4PDVars+=["tau_etOverPtLeadTrk"]
        self.tauD4PDVars+=["tau_etOverPtLeadTrk_EMJES"]
        
        self.tauD4PDVars+=["tau_calcVars_topoInvMass"]
        self.tauD4PDVars+=["tau_calcVars_topoInvMass_EMJES"]
        
        self.tauD4PDVars+=["tau_calcVars_emFracCalib"]
        self.tauD4PDVars+=["tau_calcVars_emFracCalib_EMJES"]

        self.tauD4PDVars+=["tau_massTrkSys"]
        self.tauD4PDVars+=["tau_trkWidth2"]
        self.tauD4PDVars+=["tau_author"]
        self.tauD4PDVars+=["tau_BDTJetScore"]
        self.tauD4PDVars+=["tau_likelihood"]
        self.tauD4PDVars+=["tau_ipZ0SinThetaSigLeadTrk"]
        self.tauD4PDVars+=["tau_leadTrkPt"]
        self.tauD4PDVars+=["tau_ipSigLeadTrk"]
        self.tauD4PDVars+=["tau_ipSigLeadLooseTrk"]
        self.tauD4PDVars+=["tau_leadLooseTrkPt"]
        self.tauD4PDVars+=["tau_chrgLooseTrk"]
        self.tauD4PDVars+=["tau_trFlightPathSig"]
        self.tauD4PDVars+=["tau_etEflow"]
        self.tauD4PDVars+=["tau_mEflow"]
        self.tauD4PDVars+=["tau_seedCalo_etEMAtEMScale"]
        self.tauD4PDVars+=["tau_seedCalo_etHadAtEMScale"]
        self.tauD4PDVars+=["tau_seedCalo_etEMCalib"]
        self.tauD4PDVars+=["tau_seedCalo_etHadCalib"]
        self.tauD4PDVars+=["tau_seedCalo_trkAvgDist"]
        self.tauD4PDVars+=["tau_seedCalo_hadRadius"]
        self.tauD4PDVars+=["tau_seedCalo_centFrac"]
        self.tauD4PDVars+=["tau_seedCalo_EMRadius"]
        self.tauD4PDVars+=["tau_seedCalo_isolFrac"]
        self.tauD4PDVars+=["tau_seedCalo_eta"]
        self.tauD4PDVars+=["tau_seedCalo_phi"]
        self.tauD4PDVars+=["tau_seedTrk_EMRadius"]
        self.tauD4PDVars+=["tau_calcVars_effTopoMeanDeltaR"]
        self.tauD4PDVars+=["tau_calcVars_effTopoInvMass"]
        self.tauD4PDVars+=["tau_calcVars_numEffTopoClusters"]
        self.tauD4PDVars+=["tau_calcVars_topoMeanDeltaR"]
        
        # separate list for ints
        self.tauD4PDVarsInt=["tau_numTrack"]
        self.tauD4PDVarsInt+=["tau_seedCalo_nIsolLooseTrk"]
        self.tauD4PDVarsInt+=["tau_calcVars_numTopoClusters"]
        self.tauD4PDVarsInt+=["tau_seedTrk_nIsolTrk"]
        self.tauD4PDVarsInt+=["tau_seedCalo_nStrip"]
        self.tauD4PDVarsInt+=["tau_nPi0"]
        self.tauD4PDVarsInt+=["tau_nProngLoose"]
        self.tauD4PDVarsInt+=["tau_nLooseTrk"]
        self.tauD4PDVarsInt+=["tau_nLooseConvTrk"]
        self.tauD4PDVarsInt+=["tau_cell_n"]
        self.tauD4PDVarsInt+=["tau_track_n"]
        # calculated variables about taus
        self.tauD4PDVarsCalcInt=["tau_n"]
        self.tauD4PDVarsCalcInt+=["tau_intAuthor"]
        self.tauD4PDVarsCalc=["weight"]
        # truth
        if self.doTruth:
            self.tauD4PDVarsInt+=["tau_trueTauAssocSmall_index"]
            self.tauD4PDVarsCalcInt+=["tau_numProngsOfMatch"]
            self.tauD4PDVarsCalcInt+=["tau_isTruthMatched"]
            self.tauD4PDVarsCalc+=["tau_EtVisOfMatch"]
            self.tauD4PDVarsCalc+=["tau_EtaVisOfMatch"]
        # non-tau variables to be filled per tau (all ints)
        self.nonTauVarsPerTauInt=["lbn"]
        self.nonTauVarsPerTauInt+=["RunNumber"]
        self.nonTauVarsPerTauInt+=["EventNumber"]
        self.nonTauVarsPerTauInt+=["L1_J5"]
        self.nonTauVarsPerTauInt+=["L1_TAU5"]
        # need some variables to be read for event filtering
        self.filterVars=["jet_isGood"]
        self.filterVars+=["vxp_nTracks"]
        self.filterVars+=["tau_cluster_E"]
        self.filterVars+=["tau_cluster_eta"]
        self.filterVars+=["tau_cluster_phi"]
        print "End Load Variables"
    
    #__________________________________________________________________
    def LoadHistograms(self):
    
        self.Histograms = {}
        self.LoadVariables()
        # ntuple per tau
        self.D4PD = ROOT.TTree('D4PD','flattened D3PD')
        for tauVar in self.tauD4PDVars+self.tauD4PDVarsCalc:
            self.branches[tauVar] = array.array('f',[0])
            self.D4PD.Branch(tauVar,self.branches[tauVar],tauVar+'/F')
        for tauVar in self.tauD4PDVarsInt+self.tauD4PDVarsCalcInt+self.nonTauVarsPerTauInt:
            self.branches[tauVar] = array.array('i',[0])            
            self.D4PD.Branch(tauVar,self.branches[tauVar],tauVar+'/I')
        self.GetOutputList().Add( self.D4PD )
        # ntuple per true tau
        if self.doTruth:
            self.D4PDTruth = ROOT.TTree('D4PDTruth','true taus')
            trueVar='trueTau_vis_eta'
            self.branches[trueVar] = array.array('f',[0])
            self.D4PDTruth.Branch(trueVar,self.branches[trueVar],trueVar+'/F')
            trueVar='trueTau_vis_Et'
            self.branches[trueVar] = array.array('f',[0])
            self.D4PDTruth.Branch(trueVar,self.branches[trueVar],trueVar+'/F')
            trueVar='trueTau_nProng'
            self.branches[trueVar] = array.array('i',[0])
            self.D4PDTruth.Branch(trueVar,self.branches[trueVar],trueVar+'/I')
            trueVar='trueTau_etOfMatch'
            self.branches[trueVar] = array.array('f',[0])
            self.D4PDTruth.Branch(trueVar,self.branches[trueVar],trueVar+'/F')
            #
            self.GetOutputList().Add( self.D4PDTruth )
        print "End Load Histograms"
    
    #__________________________________________________________________
    def InitializeEventStore(self):
    
        print "Begin InitializeEventStore"
        
        # protect people against asking for variables which don't exist
        # in input tree. Build outputTreeList.
        self.outputTreeList=[]
        #
        if self.tree == None:
            return
        self.LoadVariables()
        # Get list of branches in current tree
        nBranches = self.tree.GetListOfBranches().GetEntries()
        branchList=[]
        for ibranch in xrange(nBranches):
            branchList.append(self.tree.GetListOfBranches()[ibranch].GetName())
        # turn-on reading only of branches we need
        self.tree.SetBranchStatus( "*", False )
        for var in self.tauD4PDVars+self.tauD4PDVarsInt+self.nonTauVarsPerTauInt:
            if var in branchList:
                self.tree.SetBranchStatus(var, True)
                self.outputTreeList.append(var)
        # Some variables need to be turned on for the event filtering
        for var in self.filterVars:
            if var in branchList:
                self.tree.SetBranchStatus(var, True)
                self.outputTreeList.append(var)
        # Always read run number. Needed for metaData
        self.runNumber = array.array('i',[0 ])
        self.tree.SetBranchStatus( "RunNumber", True ) 
        self.tree.SetBranchAddress( "RunNumber", self.runNumber )     
        self.outputTreeList.append("RunNumber")
        #
        if self.doTruth:
            self.tree.SetBranchStatus( "tau_trueTauAssocSmall_index", True ) 
            self.tree.SetBranchStatus( "trueTau_tauAssocSmall_index", True )
            self.tree.SetBranchStatus( "trueTau_nProng", True ) 
            self.tree.SetBranchStatus( "trueTau_vis_Et", True ) 
            self.tree.SetBranchStatus( "trueTau_vis_eta", True ) 

        print "Done InitializeEventStore"

    #__________________________________________________________________        
    def LoadMetadata(self):
    
        if self.runNumber[0]>152165: # 7TeV data!!!
            self.currentSampleWeight = 1.0
            return
        # determine if this run number is MC OR DATA
        currentSample = None
        for key, value in datasets.MC7TeV.items()+datasets.Data7TeV.items():
            if value.runnumber == self.runNumber[0]:
                currentSample = key
                break
        s = currentSample
        if s == None:
            print 'ERROR: can\'t locate dataset for [',str(self.runNumber[0]),'] in module'
            self.Abort( 'ERROR' )
        # Weight is just cross section given in datasets file
        # you can pre-divide that number by number of events if you want
        # doing it here will not work since proof only knows the number
        # of events given to its processor... 
        if s in datasets.Data7TeV.keys():
            self.currentSampleWeight = datasets.Data7TeV[s].sigma
        else:
            self.currentSampleWeight = datasets.MC7TeV[s].sigma

