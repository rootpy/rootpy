import numpy as np

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
