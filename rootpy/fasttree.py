import numpy as np
from rootpy.basictypes import *

class FastTree:
    
    def __init__(self, trees, branchNames=None):
        
        if branchNames != None:
            if type(branchNames) is not list:
                branchNames = [branchNames]
        self.specialBranchNames = ["__weight"]
        self.branchNames = branchNames
        
        if self.branchNames == None: 
            self.branchNames = [branch.GetName() for branch in trees[0].GetListOfBranches()]
        branches = dict([(name, []) for name in self.branchNames + self.specialBranchNames])
        buffer = dict([(name, Float()) for name in self.branchNames])
        
        #read in trees as lists
        reader = TreeReader(trees, buffer)
        while reader.read():
            for name in self.branchNames:
                branches["__weight"].append(reader.weight)
                branches[name].append(buffer[name].value())
        
        #convert to numpy array
        self.arrays = dict([(name, np.array(branches[name])) for name in self.branchNames])
    
    def sort(self, branch):

        if self.arrays.has_key(branch):
            inx = np.argsort(self.arrays[branch])
            for key in self.arrays.keys():
                self.arrays[key] = np.array([self.arrays[key][i] for i in inx])
    
    def getListOfBranches(self):
        
        return self.arrays.keys()
    
    def getBranch(self, name):
        
        if self.arrays.has_key(name):
            return self.arrays[name]
        return None
    
    """
    def apply_cut(self, name, low=None, high=None):
        
        if name not in self.branchToIndex.keys():
            return
        index = self.branchToIndex[name]
        if low != None and high != None:
            condition = (self.crop[index] >= low) & (self.crop[index] < high)
        elif low != None:
            condition = self.crop[index] >= low
        elif high != None:
            condition = self.crop[index] < high
        else:
            return
        self.crop = self.crop.compress(condition, axis=1)
    
    def apply_cuts(self, cuts):
        
        self.reset()
        for cut in cuts:
            self.apply_cut(cut["variable"], low=cut["low"], high=cut["high"])
    """


class TreeReader:
    
    def __init__(self, treeList, branchMap, branchList=None, subs=None):
        
        if type(treeList) is not list:
            treeList = [treeList]
        assert(len(treeList)>0)
        self.treeList = [tree for tree in treeList]
        self.branchMap = branchMap
        self.subs = subs
        
        if not branchList:
            self.branchList = self.branchMap.keys()
        else:
            self.branchList = branchList
            
        self.weight = 1.
        self.tree = None
        self.entry = 0
        self.entries = 0
        
    def initialize(self):

        if self.tree != None:
            self.tree.ResetBranchAddresses()
        if len(self.treeList) > 0:
            self.tree = self.treeList.pop()
            self.entry = 0
            self.entries = self.tree.GetEntries()
            for branch in self.branchList:
                subBranch = branch
                if self.subs:
                    if branch in self.subs.keys():
                        subBranch = self.subs[branch]
                if not self.tree.GetBranch(subBranch):
                    raise RuntimeError("Branch %s was not found in tree %s"%(subBranch,self.tree.GetName()))
                self.tree.SetBranchAddress(subBranch,self.branchMap[branch])
            return True
        return False
    
    def isReady(self):
        
        return self.entry < self.entries
    
    def read(self):
        
        if not self.isReady():
            if not self.initialize():
                return False
        self.tree.GetEntry(self.entry)
        self.weight = self.tree.GetWeight()
        self.entry += 1
        return True

