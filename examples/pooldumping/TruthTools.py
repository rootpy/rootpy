import PyCintex
import ROOT

truthHandler = ROOT.TruthHandler()
    
def numProng(trueTau):

    return truthHandler.nProngTruth(trueTau,True)
    
def getTrueHadronicTaus(truthParticlesContainer):
    
    return truthHandler.getHadronicTruth(truthParticlesContainer)

def getTrueTauVisibleSum(trueTau):

    return truthHandler.getTauVisibleSumTruth(trueTau)
