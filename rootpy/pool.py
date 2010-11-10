import PyCintex
import ROOT
import AthenaROOTAccess.transientTree
from RootUtils.PyROOTFixes import enable_tree_speedups
enable_tree_speedups()

class Event:

    def __init__(self, index, containers):
        
        self.index = index
        for name,container in containers.items():
            setattr(self,name,container)
            if name == "EventInfo":
                self.id = self.EventInfo.event_ID()
                self.runNumber = self.id.run_number()
                self.lumiBlock = self.id.lumi_block()
                self.number = self.id.event_number()

class EventStream:
    
    def __init__(self,fileList,branchNames,numEvents=-1):
        
        self.branchNames = branchNames
        self.containers = {}
        self.branches = []
        if type(fileList) is not list:
            self.fileList = [fileList]
        else:
            self.fileList = fileList
        self.file = None
        self.tree = None
        self.index = 0
        self.eventNumber = 1
        self.maxNumEvents = numEvents
        self.events = 0
        print "EventStream: dumping from %i files"%(len(self.fileList))
        self.initialize()
         
    def initialize(self):

        if self.file != None:
            self.file.Close()
            
        if len(self.fileList) > 0:
             
            self.file = ROOT.TFile.Open(self.fileList.pop(0))
            self.tree = AthenaROOTAccess.transientTree.makeTree(self.file, branchNames = self.branchNames)
            ROOT.SetOwnership(self.tree, False)
            self.events = self.tree.GetEntries()
            self.index = 0
            self.containers = {}
            self.branches = []
            for branch in self.branchNames.values():
                self.containers[branch] = eval("self.tree."+branch)
                self.branches.append(self.tree.GetBranch(branch))
            return True
        return False
 
    def read(self):

        if self.eventNumber == self.maxNumEvents + 1:
            if self.file != None:
                self.file.Close()
            return None
        
        if self.index == self.events:
            if not self.initialize():
                return None
        
        for branch in self.branches:
            branch.GetEntry(self.index)
        
        currEvent = Event(self.index,self.containers)
        self.eventNumber += 1
        self.index += 1
        return currEvent
