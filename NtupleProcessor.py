from NtupleBuffer import *

demote = {"VECTOR<FLOAT>":"F",
          "VECTOR<INT>":"I",
          "VF","F",
          "VI":"I"}

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
        self.inBuffer.fuse(self.inTree)
        self.outBuffer.fuse(self.outTree)
        self.entries = self.inTree.GetEntries()
        self.entry = 0
        self.flatten = flatten

    def next(self):

        if self.entry < self.entries:
            self.inTree.GetEntry(self.entry)
            return True
        return False

    def copy(self):

        if self.flatten:
            while self.next():
                self.outTree.Fill()
        else:
            while self.next():
                self.outTree.Fill()
