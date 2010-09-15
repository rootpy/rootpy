from NtupleBuffer import *

# inTree is an existing tree containing data (entries>0).
# outTree is a new tree, not necessarily containing any branches, and should not contain any data (entries==0).
class NtupleProcessor(object):

    def __init__(self,inTree,outTree,inVars=None,outVars=None,flatten=False):

        self.inTree = inTree
        self.outTree = outTree
        self.inVars = inVars
        if not self.inVars:
            self.inVars = [(branch.GetName(),branch.GetListOfLeaves()[0].GetTypeName().upper()) for branch in inTree.GetListOfBranches()]
        self.outVars = outVars
        if not self.outVars:
            self.outVars = self.inVars
        self.inBuffer = NtupleBuffer(self.inVars)
        self.outBuffer = self.inBuffer
        self.inBuffer.fuse(self.inTree,createMissing=False)
        self.outBuffer.fuse(self.outTree,createMissing=True)
        self.entries = self.inTree.GetEntries()
        self.entry = 0
        self.flatten = flatten

    def read(self):

        if self.entry < self.entries:
            self.inTree.GetEntry(self.entry)
            return True
        return False

    def write(self):

        self.outTree.Fill()

    def copy(self):

        if self.flatten:
            while self.next():
                self.write()
        else:
            while self.next():
                self.write()
