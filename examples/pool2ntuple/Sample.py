import ROOT
import TauVariables
from Ntuple import Ntuple
from pyTauDetailsManager import pyTauDetailsManager
import pyHepMC
import PDG
from Jet import Jet
import TruthTools
import EventFilter

class Sample:
    
    def __init__(self,name,outputFile):
        
        self.name = name
        self.outputFile = outputFile
        self.totalEvents = 0
        self.details = pyTauDetailsManager()
        self.recoARAVariables = TauVariables.getVarMap(["binning","variables"])
        self.recoMap = TauVariables.getVarMap(["binning","variables","RecoJet","flags","discriminants"])
        self.flags = TauVariables.getVarMap(["flags"])
        self.outputFile.cd()
        self.recoTuple = Ntuple(self.name,self.recoMap)
        self.eventTuple = Ntuple(self.name+"_EventInfo",self.flags)
        
    def finalize(self): pass
                 
    def fill(self,event):pass
    
    def log(self,message):
        
        print message

class RealSample(Sample):
    
    def __init__(self,name,outputFile):
        
        Sample.__init__(self,name,outputFile)
    
    def finalize(self):

        self.outputFile.cd()
        self.recoTuple.write()
        self.eventTuple.write()
    
    def fill(self,event,verbose=False):
        
        if not EventFilter.passes(event):
            return

        if verbose: self.log(str(len(event.RecoJets)) + " reco jets")

        self.flags["index"].set(self.totalEvents)
        self.flags["run"].set(event.runNumber)
        self.flags["lumiblock"].set(event.lumiBlock)
        self.flags["event"].set(event.number)

        self.eventTuple.fill()
        
        for recoJet in event.RecoJets:
            if not self.details.update(recoJet):
                self.log("Updating tau details failed!")
            else:
                for name,variable in self.recoARAVariables.items():
                    if variable.type() == 'F':
                        variable.set(self.details.getFloatDetailValue(name.upper()))
                    else:
                        variable.set(self.details.getIntDetailValue(name.upper()))
                self.recoMap["likelihood"].set(recoJet.tauID().likelihood())
                self.recoMap["BDTJetScore"].set(recoJet.tauID().BDTJetScore())
                self.recoTuple.fill()
        
        self.totalEvents += 1

class MCSample(Sample):
    
    def __init__(self,name,type,outputFile,generator = None,verbose = False):
        
        Sample.__init__(self,name,outputFile)
        self.type = type
        if type == "TAU":
            self.truthMap = TauVariables.getVarMap(["TruthTau","flags"])
        else:
            self.truthMap = TauVariables.getVarMap(["TruthJet","flags"])
        self.truthTuple = Ntuple("_".join([self.name,"truth"]),self.truthMap)
        
        self.matcher = ROOT.Matcher("DataVector<INavigable4Momentum>,DataVector<INavigable4Momentum>,I4MomentumDeltaR,I4MomentumCompare")(verbose)
        
        self.jetFilter = ROOT.TruthFilter()
        if generator == "PYTHIA":
            self.priVertex = -5
            self.jetFilter.setAcceptStatuses("3")
        else:
            self.priVertex = -1
            self.jetFilter.setAcceptStatuses("141,142,143,144")
        pdgs = ",".join([str(a) for a in (range(-5,0) + range(1,6) + [21])])
        self.jetFilter.setAcceptPDGIDs(pdgs)

        self.elecFilter = ROOT.TruthFilter()
        self.elecFilter.setAcceptStatuses("3")
        self.elecFilter.setAcceptPDGIDs("11,-11")

        self.generator = generator
    
    def printParticle(self,particle):
        
        return "%6s\teta:% .2f\tphi:% .2f\tpT: %.2f"%(PDG.pdgid_to_name(particle.pdg_id()),particle.momentum().eta(),particle.momentum().phi(),particle.momentum().perp())
    
    def finalize(self):

        self.outputFile.cd()
        self.recoTuple.write()
        self.truthTuple.write()
        self.eventTuple.write()
    
    def fill(self,event,verbose=False):
        
        for map in [self.recoMap,self.truthMap,self.flags]:
            for key,value in map.items():
                value.reset()

        self.flags["index"].set(self.totalEvents)
        self.flags["event"].set(event.number)

        self.eventTuple.fill()

        if self.type == "TAU":
            trueJets = TruthTools.getTrueHadronicTaus(event.TruthParticles)
            if verbose: self.log("%i hadronic MC taus"%len(trueJets))
        elif self.type == "ELEC":
            trueJets = self.elecFilter.toTruthParticleContainer(self.elecFilter.filter(event.HepMC))
            if verbose: self.log("%i MC electrons"%len(trueJets))
        else:
            trueJets = event.TruthJets
            if verbose: self.log("%i MC jets:"%len(trueJets))
        
        if verbose: self.log("%i tau candidates"%len(event.RecoJets))
        
        # Matching truthJets to recoJets
        self.matcher.matchGreedy(trueJets,event.RecoJets,0.2,False)
        matches = self.matcher.getMap()
        if verbose: self.log("found %i matches"%matches.size())
        
        decoratedTrueJets = [Jet(jet,self.type) for jet in self.matcher.getUnmatchedLeft()]

        decoratedRecoJets = [Jet(jet,"RECO") for jet in self.matcher.getUnmatchedRight()]

        for jet in self.matcher.getMatchedLeft():
            decorTrueJet = Jet(jet,self.type)
            matchedJet = matches[jet][0]
            decorRecoJet = Jet(matchedJet,"RECO")
            decorRecoJet.setMatch(decorTrueJet)
            decorTrueJet.setMatch(decorRecoJet)
            decoratedRecoJets.append(decorRecoJet)
            decoratedTrueJets.append(decorTrueJet)
            if verbose: self.log("%s --> %s"%(decorTrueJet,decorRecoJet))

        if self.type == "JET" and self.generator:
            genParticles = self.jetFilter.toTruthParticleContainer(self.jetFilter.filterFromVertex(event.HepMC,self.priVertex))
            if verbose: print "%i generator particles remain after filtering"%len(genParticles)
            if genParticles:
                self.matcher.matchLazyGreedy(event.RecoJets,genParticles,0.6,False)
                matchMap = self.matcher.getMap()
                for recoJet in decoratedRecoJets:
                    matchedList = self.matcher.lookup(recoJet.jet)
                    if matchedList:
                        recoJet.pdgid = abs(matchedList[0].pdgId())

        for trueJet in decoratedTrueJets:
            self.truthMap["etaVis"].set(trueJet.eta)
            self.truthMap["eVis"].set(trueJet.e)
            self.truthMap["etVis"].set(trueJet.et)
            self.truthMap["ptVis"].set(trueJet.pt)
            self.truthMap["matchesReco"].set(int(trueJet.match != None))
            if trueJet.match != None:
                self.truthMap["eOfMatch"].set(trueJet.match.e)
                self.truthMap["etOfMatch"].set(trueJet.match.et)
                self.truthMap["ptOfMatch"].set(trueJet.match.pt)
            if self.type == "TAU":
                self.truthMap["numProng"].set(trueJet.numTracks)
                if trueJet.match != None:
                    self.truthMap["numTracksOfMatch"].set(trueJet.match.numTracks)
            else:
                self.truthMap["pdgid"].set(trueJet.pdgid)
            self.truthTuple.fill()
       
        pdgNames = []
        for recoJet in decoratedRecoJets:
            if not self.details.update(recoJet.jet):
                self.log("Updating tau details failed!")
            else:
                for name,variable in self.recoARAVariables.items():
                    if variable.type() == 'F':
                        variable.set(self.details.getFloatDetailValue(name.upper()))
                    else:
                        variable.set(self.details.getIntDetailValue(name.upper()))
                self.recoMap["pdgid"].set(recoJet.pdgid)
                if recoJet.pdgid > 0:
                    pdgNames.append(PDG.pdgid_to_name(recoJet.pdgid))
                self.recoMap["matchesTruth"].set(int(recoJet.match!=None))
                if recoJet.match != None:
                    self.recoMap["numProngsOfMatch"].set(recoJet.match.numTracks)                
                    self.recoMap["eVisOfMatch"].set(recoJet.match.e)
                    self.recoMap["etVisOfMatch"].set(recoJet.match.et)
                    self.recoMap["ptVisOfMatch"].set(recoJet.match.pt)
                    self.recoMap["etaVisOfMatch"].set(recoJet.match.eta)
                self.recoMap["likelihood"].set(recoJet.jet.tauID().likelihood())
                self.recoMap["BDTJetScore"].set(recoJet.jet.tauID().BDTJetScore())
                self.recoMap["ElectronVetoLoose"].set(recoJet.jet.tauID().electronVetoLoose())
                self.recoMap["ElectronVetoMedium"].set(recoJet.jet.tauID().electronVetoMedium())
                self.recoMap["ElectronVetoTight"].set(recoJet.jet.tauID().electronVetoTight())
                self.recoTuple.fill()
        if len(pdgNames) > 0 and verbose:
            print "reco taus have been matched to:"
            print " ".join(pdgNames)
        self.totalEvents += 1
