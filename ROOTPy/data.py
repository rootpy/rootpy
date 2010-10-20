import ROOT
import DataLibrary
from ROOTPy.ntuple import Cut
import uuid
import os
from array import array

sampleSets = {"default" : Cut(""),
              "train"   : Cut("event%2==0"),
              "test"    : Cut("event%2==1")}

class DataManager:
    
    def __init__(self,cache=True,verbose=False):
        
        self.verbose = verbose
        self.docache = cache
        self.coreData = None
        self.coreDataName = None
        self.pluggedData = None
        self.pluggedDataName = None
        self.accessHistory = {}
        self.friendFiles = {}
        self.scratchFileName = uuid.uuid4().hex+".root"
        self.scratchFile = ROOT.TFile.Open(self.scratchFileName,"recreate")
    
    def __del__(self):
        
        if self.scratchFile:
            self.scratchFile.Close()
            os.remove(self.scratchFileName)
        if self.coreData:
            self.coreData.Close()
        if self.pluggedData:
            self.pluggedData.Close()
        for file in self.friendFiles.values():
            file.Close()
    
    def load(self,filename):
        
        if self.verbose: print "loading %s"%filename
        data = ROOT.TFile.Open(filename)
        if data:
            if self.coreData:
                self.removeFromAccessHistory(self.coreDataName)
                self.coreData.Close()
            self.coreData = data
            self.coreDataName = filename
            self.accessHistory[filename] = {}
        else:
            print "Could not open %s"%filename
    
    def plug(self,filename):
       
        if not self.coreData:
            print "Cannot plug in supplementary data with no core data!"
            return
        if not filename:
            if self.pluggedData:
                self.removeFromAccessHistory(self.pluggedDataName)
                self.pluggedData.Close()
            self.pluggedData = None
            self.pluggedDataName = None
        else:
            if self.verbose: print "plugging in %s"%filename
            data = ROOT.TFile.Open(filename)
            if data:
                if self.pluggedData:
                    self.removeFromAccessHistory(self.pluggedDataName)
                    self.pluggedData.Close()
                self.pluggedData = data
                self.pluggedDataName = filename
                self.accessHistory[filename] = {}
            else:
                print "Could not open %s"%filename
    
    def removeFromAccessHistory(self,filename):
        
        del self.accessHistory[filename]

    def clear_cache(self):
        
        self.accessHistory = {}
        if self.pluggedDataName != None:
            self.accessHistory[self.pluggedDataName] = {}
        if self.coreDataName != None:
            self.accessHistory[self.coreDataName] = {}
         
    def getObjectFromFiles(self,name):
        
        for file,filename in [(self.pluggedData,self.pluggedDataName), (self.coreData,self.coreDataName)]:
            if file:
                object = file.Get(name)
                if object:
                    return (object,filename)
        return (None,None)
            
    def getWeight(self,tree,sampleid,sampleType="default",fraction=1.): 
        
        assert(fraction>0)
        name = DataLibrary.xSectionDict7TeV[sampleid]["name"]
        xsec = DataLibrary.xSectionDict7TeV[sampleid]["xsec"]
        if xsec < 0.:
            if self.verbose: print "Sample is listed with a negative cross-section."
            if not tree:
                if self.verbose: print "Returning a weight of 1."
                return 1.
            weightBranch = tree.GetBranch("weight")
            if weightBranch:
                if self.verbose: print "Will use the weight branch to determine tree weight. Assuming the weight branch is constant!"
                buffer = array('f',[0.])
                weightBranch.SetAddress(buffer)
                weightBranch.GetEntry(0)
                weight = buffer[0]
                if self.verbose: print "Returning a weight of %f."%weight
                tree.ResetBranchAddresses()
                return weight
            if self.verbose: print "Returning a weight of 1."
            return 1.
        sampleCuts = sampleSets[sampleType]
        eventInfoName = "_".join([name,"EventInfo"])
        if self.verbose:
            print "Calculating weight for sample %s with xsection %f..."%(name,xsec)
            print "looking for %s"%eventInfoName
        eventInfo,filename = self.getObjectFromFiles(eventInfoName)
        if not eventInfo:
            print "Error: unable to find %s. Returning -1 as weight."%eventInfoName
            return -1.
        origNumEvents = eventInfo.GetEntries()
        if not sampleCuts.empty():
            if self.verbose: print "applying cuts %s"%sampleCuts
            eventInfo = eventInfo.CopyTree(str(sampleCuts))
        numEvents = eventInfo.GetEntries()
        if self.verbose: print "using %i of %i events to determine weight"%(numEvents,origNumEvents)
        if numEvents < 1:
            print "Error: zero entries for sample %s. "%name
            return -1.
        if fraction != 1.:
            print "scaling weight inversely by the fraction of events used: %f"%fraction
        weight = xsec/(numEvents*fraction)
        if self.verbose: print "Sample has weight of %e"%weight
        return weight

    def normalizeWeights(self,trees,norm=1.):
        
        totalWeight = 0.
        for tree in trees:
            totalWeight += tree.GetWeight()
        for tree in trees:
            tree.SetWeight(norm*tree.GetWeight()/totalWeight)
    
    def getTree(self, treeName, sampleid=None, truth=False, maxEntries=-1, fraction=-1, cuts=None, sampleType="default"):
        
        if not cuts:
            cuts = Cut("")
        if sampleType:
            cuts *= sampleSets[sampleType]
        if truth:
            treeName+="_truth"
        if self.verbose: print "Fetching tree %s..."%treeName
        treeName_temp = treeName
        if not cuts.empty():
            treeName_temp+=":"+str(cuts)
        inFile = None
        if self.docache:
            for filename in self.accessHistory.keys():
                if treeName_temp in self.accessHistory[filename].keys():
                    inFile = filename
                    break
        filename = ""
        tmpTree = None
        if inFile != None and self.docache:
            if self.verbose: print "%s is already in memory and was loaded from %s"%(treeName_temp,inFile)
            tree = self.accessHistory[inFile][treeName_temp]
        else:
            tree,filename = self.getObjectFromFiles(treeName)
            if not tree:
                print "Tree %s not found!"%treeName
                return None
            friends = tree.GetListOfFriends()
            if friends:
                if len(friends) > 0 and self.verbose:
                    print "Warning! ROOT does not play nice with friends where cuts are involved!"
                if len(friends) == 1:
                    if self.verbose:
                        print "Since this tree has one friend, I will assume that it's friend is the core data (read-only) and you want that tree instead"
                        print "where cuts may refer to branches in this tree"
                    tmpTree = tree
                    friendTreeName = friends[0].GetTreeName()
                    if self.friendFiles.has_key(friends[0].GetTitle()):
                        friendTreeFile = self.friendFiles[friends[0].GetTitle()]
                    else:
                        friendTreeFile = ROOT.TFile.Open(friends[0].GetTitle())
                        self.friendFiles[friends[0].GetTitle()] = friendTreeFile
                    filename = friends[0].GetTitle()
                    tree = friendTreeFile.Get(friendTreeName)
                    tree.AddFriend(tmpTree)
                elif len(friends) > 1 and self.verbose:
                    print "Warning! This tree has multiple friends!"
            if not cuts.empty():
                print "Applying cuts %s"%cuts
                if friends:
                    if len(friends) > 1 and self.verbose:
                        print "Warning: applying cuts on tree with multiple friends is not safely implemented yet"
                self.scratchFile.cd()
                tree = tree.CopyTree(str(cuts))
        originalNumEntries = tree.GetEntries()
        if fraction > -1.:
            entries = tree.GetEntries()
            if self.verbose: print "Extracting %.1f%% of the tree which contains %i entries."%(fraction*100.,entries)
            newEntries = int(fraction*entries)
            self.scratchFile.cd()
            tree = tree.CloneTree(newEntries)
        elif maxEntries > -1 and tree.GetEntries() > maxEntries:
            if self.verbose:
                print "Number of entries in tree exceeds maximum allowed by user: %i"%maxEntries
                print "Extracting %i of %i total entries"%(maxEntries,tree.GetEntries())
            self.scratchFile.cd()
            tree = tree.CloneTree(maxEntries)
        finalNumEntries = tree.GetEntries()
        if sampleid:
            if finalNumEntries == 0:
                tree.SetWeight(self.getWeight(tree,sampleid,sampleType=sampleType,fraction=1.))
            else:
                tree.SetWeight(self.getWeight(tree,sampleid,sampleType=sampleType,fraction=float(finalNumEntries)/float(originalNumEntries)))
        if self.verbose: print "Found %s with %i entries and weight %e"%(treeName,tree.GetEntries(),tree.GetWeight())
        if inFile == None:
            tree.SetName(treeName_temp)
            if self.docache:
                if self.verbose: print "Keeping reference to %s in %s"%(tree.GetName(),filename)
                if not self.accessHistory.has_key(filename):
                    self.accessHistory[filename] = {}
                self.accessHistory[filename][tree.GetName()] = tree
                if tmpTree != None:
                    self.accessHistory[filename]["%s==>%s"%(tree.GetName(),tmpTree.GetName())] = tmpTree
        return tree
    
    def getSample(self,samples,truth=False,maxEntries=-1,fraction=-1,sampleType="default"):
        
        assert(sampleType in sampleSets.keys())
        sampleList = []
        for sample in samples:
            if self.verbose: print "==========================================================="
            sampleName = sample["name"]
            sampleWeight = sample["weight"]
            sampleJetType = sample["jetType"]
            objectCuts = sample["objCuts"]
            cuts = Cut("")
            if objectCuts:
                objectCuts.setJetType(sampleJetType)
                if truth:
                    cuts=objectCuts.getTruthCut()
                else:
                    cuts=objectCuts.getRecoCut()
            if sampleName not in DataLibrary.sampleGroupsDict.keys():
                print "Unknown sample requested!"
                return None
            subsamples = []
            for sampleid in DataLibrary.sampleGroupsDict[sampleName]["idlist"]:
                if truth:
                    subsampleName = DataLibrary.xSectionDict7TeV[sampleid]["name"]
                else:
                    subsampleName = DataLibrary.xSectionDict7TeV[sampleid]["name"]
                tree = self.getTree(subsampleName,sampleid,truth=truth,maxEntries=maxEntries,fraction=fraction,cuts=cuts,sampleType=sampleType)
                if tree:
                    if sampleWeight < 0:
                        if self.verbose: print "Tree weight set to 1.0 as requested"
                        tree.SetWeight(1.)
                    subsamples.append(tree)
                else:
                    return None
                if self.verbose: print "-----------------------------------------------------------"
            if sampleWeight > 0:
                if self.verbose: print "Normalizing weights to %f"%sampleWeight
                self.normalizeWeights(subsamples, sampleWeight)
            sampleList += subsamples
        return sampleList

sampleGroupsDict = {"J0-J4"        : {"idlist":[105009,105010,105011,105012,105013],"class":0,"name":"J0-J4"},
                    "J0-J5"        : {"idlist":[105009,105010,105011,105012,105013,105014],"class":0,"name":"J0-J5"},
                    "J1-J3"        : {"idlist":[105010,105011,105012],"class":0,"name":"J1-J3"},
                    "J1-J4"        : {"idlist":[105010,105011,105012,105013],"class":0,"name":"J1-J4"},
                    "J1-J5"        : {"idlist":[105010,105011,105012,105013,105014],"class":0,"name":"J1-J5"},
                    "J2-J5"        : {"idlist":[105011,105012,105013,105014],"class":0,"name":"J2-J5"},
                    "J0"           : {"idlist":[105009],"class":0,"name":"J0"},
                    "J1"           : {"idlist":[105010],"class":0,"name":"J1"},
                    "J2"           : {"idlist":[105011],"class":0,"name":"J2"},
                    "J3"           : {"idlist":[105012],"class":0,"name":"J3"},
                    "J4"           : {"idlist":[105013],"class":0,"name":"J4"},
                    "J5"           : {"idlist":[105014],"class":0,"name":"J5"},
                    "JX_n5"        : {"idlist":[105010.5,105011.5,105012.5],"class":0,"name":"J1-J3 with pileup (n=5)"},
                    "WmunuNp0"     : {"idlist":[107690],"class":0},
                    "WmunuNp1"     : {"idlist":[107691],"class":0},
                    "WmunuNp2"     : {"idlist":[107692],"class":0},
                    "WmunuNp3"     : {"idlist":[107693],"class":0},
                    "WmunuNp4"     : {"idlist":[107694],"class":0},
                    "WmunuNp5"     : {"idlist":[107695],"class":0},
                    "Wmunu"        : {"idlist":[107690,107691,107692,107693,107694,107695],"class":0},
                    "semilepttbar" : {"idlist":[105200],"class":1,"name":"Semileptonic t#bar{t}"},
                    "hadttbar"     : {"idlist":[105204],"class":0,"name":"Hadronic t#bar{t}"},
                    "Ztautau"      : {"idlist":[106052],"class":1,"name":"Pythia Z#rightarrow#tau#tau"},
                    "Ztautau_n5"   : {"idlist":[106052.5],"class":1,"name":"Pythia Z#rightarrow#tau#tau with pileup (n=5)"},
                    "Zee"          : {"idlist":[106046],"class":0,"name":"Z#rightarrowee"},
                    "Wenu"         : {"idlist":[106043],"class":0,"name":"W#rightarrowe#nu"},
                    "Wtaunu"       : {"idlist":[107054],"class":1,"name":"W#rightarrow#tau#nu"},
                    "Atautau"      : {"idlist":[106573,109870,109871,109872,109873,109874,109875,109876,109877],"class":1,"name":"A#rightarrow#tau#tau"},
                    "data"         : {"idlist":[105000],"class":0,"name":"Data"},
                    "PythiaZtautau": {"idlist":[115000],"class":1,"name":"Pythia Z#rightarrow#tau#tau"},
                    "DWZtautau"    : {"idlist":[115000.5],"class":1,"name":"Pythia DW Z#rightarrow#tau#tau"},
                    "DW"           : {"idlist":[125000,135000,145000,155000],"class":0,"name":"Pythia DW J1-J4"},
                    "Perugia"      : {"idlist":[165000,175000,185000,195000],"class":0,"name":"Perugia 2010"},
                    "material"     : {"idlist":[265000,275000,285000,295000],"class":0,"name":"Different Material"},
                    "minbias"      : {"idlist":[105001],"class":0,"name":"Minimum Bias"}}

def getName(sample):

    return ", ".join([sampleGroupsDict[s]["name"] for s in sample.split("+")])
    
"""    
    uniqueFinalStates = []
    for s in subsamples:
        if "#rightarrow" in s:
            finalState = s.split("#rightarrow")[1]
            if finalState not in uniqueFinalStates:
                uniqueFinalStates.append(finalState)
    sampleName = ""
    for s in subsamples:
"""
""" dictionary with all 7TeV cross-sections (in nb) per sample: 'xsec' = sigma x eff."""
xSectionDict7TeV = {
                     105000:{"name":"data","xsec":-1.},
                     115000:{"name":"PythiaZtautau","xsec":-1.},
                     115000.5:{"name":"DWZtautau","xsec":-1.},
                     125000:{"name":"DWJ1","xsec":-1.},
                     135000:{"name":"DWJ2","xsec":-1.},
                     145000:{"name":"DWJ3","xsec":-1.},
                     155000:{"name":"DWJ4","xsec":-1.},
                     165000:{"name":"PerugiaJ1","xsec":-1.},
                     175000:{"name":"PerugiaJ2","xsec":-1.},
                     185000:{"name":"PerugiaJ3","xsec":-1.},
                     195000:{"name":"PerugiaJ4","xsec":-1.},
                     265000:{"name":"materialJ1","xsec":-1.},
                     275000:{"name":"materialJ2","xsec":-1.},
                     285000:{"name":"materialJ3","xsec":-1.},
                     295000:{"name":"materialJ4","xsec":-1.},
                     105001:{"name":"pythia_minbias","xsec":-1.},
                     105009:{"name":"J0_pythia_jetjet","xsec":9.8534E+06},
                     105010:{"name":"J1_pythia_jetjet","xsec":6.7803E+05},
                     105011:{"name":"J2_pythia_jetjet","xsec":4.0979E+04},
                     105012:{"name":"J3_pythia_jetjet","xsec":2.1960E+03},
                     105013:{"name":"J4_pythia_jetjet","xsec":8.7701E+01},
                     105014:{"name":"J5_pythia_jetjet","xsec":2.3483E+00},
                     105010.5:{"name":"J1_pythia_jetjet_n5","xsec":6.7803E+05},
                     105011.5:{"name":"J2_pythia_jetjet_n5","xsec":4.0979E+04},
                     105012.5:{"name":"J3_pythia_jetjet_n5","xsec":2.1960E+03},
                     106052:{"name":"PythiaZtautau","xsec":8.5402E-01},
                     106052.5:{"name":"PythiaZtautau_n5","xsec":8.5402E-01},
                     106046:{"name":"PythiaZee","xsec":8.5575E-01},
                     106043:{"name":"PythiaWenu","xsec":8.9380E+00},
                     107054:{"name":"PythiaWtaunu","xsec":8.9295E+00},
                     106573:{"name":"PythiabbAtautauMA800TB35","xsec":1.6648E-06},
                     109870:{"name":"PythiaAtautauMA120TB20","xsec":.0230548},
                     109871:{"name":"PythiaAtautauMA300TB20","xsec":.000406063},
                     109872:{"name":"PythiaAtautauMA100TB20","xsec":.0489015},
                     109873:{"name":"PythiaAtautauMA150TB20","xsec":.00907153},
                     109874:{"name":"PythiaAtautauMA200TB20","xsec":.00259947},
                     109875:{"name":"PythiaAtautauMA110TB20","xsec":.0331659},
                     109876:{"name":"PythiaAtautauMA130TB20","xsec":.0162838},
                     109877:{"name":"PythiaAtautauMA140TB20","xsec":.0121798},
                     105200:{"name":"T1_McAtNlo_Jimmy","xsec":1.4412E-01},
                     105204:{"name":"TTbar_FullHad_McAtNlo_Jimmy","xsec":1.4428E-01}
                   }
