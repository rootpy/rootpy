import ROOT
from ntuple import Cut
import datalibrary
import uuid
import os
from array import array

sampleSets = {"default" : Cut(""),
              "train"   : Cut("EventNumber%2==0"),
              "test"    : Cut("EventNumber%2==1")}

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
        #name = datalibrary.xSectionDict7TeV[sampleid]["name"]
        #xsec = datalibrary.xSectionDict7TeV[sampleid]["xsec"]
        if self.verbose: print "Sample is listed with a negative cross-section."
        if not tree:
            if self.verbose: print "Null tree. Returning a weight of 1."
            return 1.
        weightBranch = tree.GetBranch("weight")
        if weightBranch:
            if self.verbose: print "Will use the weight branch to determine tree weight. Assuming the weight branch is constant!"
            buffer = array('f',[0.])
            weightBranch.SetAddress(buffer)
            weightBranch.GetEntry(0)
            weight = buffer[0]
            if self.verbose: print "Returning a weight of %e."%weight
            tree.ResetBranchAddresses()
            return weight
        if self.verbose: print "Weight branch not found. Returning a weight of 1."
        return 1.
        """
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
        """

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
                if self.verbose: print "Tree %s not found!"%treeName
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
            if sampleName not in datalibrary.sampleGroupsDict.keys():
                print "Unknown sample requested!"
                return None
            subsamples = []
            for sampleid in datalibrary.sampleGroupsDict[sampleName]["idlist"]:
                subsampleName = datalibrary.xSectionDict7TeV[sampleid]["name"]
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


