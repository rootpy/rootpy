
class NtupleReader:
    
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
                self.tree.SetBranchAddress(subBranch,self.branchMap[branch].address())
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
