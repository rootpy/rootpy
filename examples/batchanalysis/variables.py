import ROOT
import math

def nEffClusters(clusterEnergy):
    sumE=0.
    sum2E=0.
    for icluster in xrange(len(clusterEnergy)):
        clusterE = clusterEnergy[icluster]
        sumE = sumE + clusterE
        sum2E = sum2E + clusterE*clusterE

    if len(clusterEnergy)>0:
        nEffConst = sumE*sumE/sum2E
    else:
        nEffConst = 0.
    return nEffConst

def nEffCells(cellEnergy):
    sumE=0.
    sum2E=0.
    for icell in xrange(len(cellEnergy)):
        cellE = cellEnergy[icell]
        sumE = sumE + cellE
        sum2E = sum2E + cellE*cellE

    if len(cellEnergy)>0:
        nEffConst = sumE*sumE/sum2E
    else:
        nEffConst = 0.
    return nEffConst

def getClusters(energies,etas,phis,energyScale=1.):

    clusters = []
    for E,eta,phi in zip(energies,etas,phis):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiE(energyScale*E/math.cosh(eta),eta,phi,E*energyScale)
        clusters.append(vector)
    # Sort descending by E
    clusters.sort(key=lambda x: x.E(), reverse=True)
    return clusters

def topoClusterMass(clusters):

    sum = ROOT.TLorentzVector()
    for cluster in clusters:
        sum = sum + cluster
    return sum.M()

def effMass(nEffConst,clusterEnergy,clusterPseudoR,clusterAzim):
    from ROOT import TLorentzVector
    from math import cosh
    clusterSum = TLorentzVector()
    oneCluster = TLorentzVector()
    # cluster list is not sorted by energy
    # no idea if this is a good way to sort!!!
    myclusterE = list(clusterEnergy)
    tmpClusterE=sorted(clusterEnergy,reverse=True)
    for icluster in xrange(int(round(nEffConst))):
        # hiIndex is the index in the original eneregy, eta, phi 
        # lists which correspond to the sorted highest E clusters
        hiIndex=myclusterE.index(tmpClusterE[icluster])
        clusterE = clusterEnergy[hiIndex]
        clusterEta = clusterPseudoR[hiIndex]
        clusterPhi = clusterAzim[hiIndex]
        clusterEt = clusterE/cosh(clusterEta)
        oneCluster.SetPtEtaPhiE(clusterEt,clusterEta,clusterPhi,clusterE)
        clusterSum = clusterSum+oneCluster
    return clusterSum.M()

def caloRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh,layerList):
    from math import cosh, fabs, fmod, sqrt
    calorad = 0.
    sumET=0.
    pi=3.141592653589793 
    for icell in xrange(len(cellEta)):
        deltaEta = fabs(jetEta - cellEta[icell])
        dPhi = jetPhi - cellPhi[icell]
        deltaPhi = fmod( dPhi+3*pi,2*pi )-pi        
        dR = sqrt( deltaEta*deltaEta + deltaPhi*deltaPhi )

        cellEnergy = cellE[icell]
        cellET = cellE[icell]/cosh(cellEta[icell])
        cellSamp = cellSampling[icell]
        if cellSamp in layerList:
            # include energy threshold possibility/cell
            if dR<0.4 and cellET>eThresh:
                calorad = calorad + dR*cellET
                sumET = sumET + cellET  
        # end cell loop
    if fabs(sumET)>0.000001:
        calorad = calorad/sumET
    if calorad==0.0:
        return -1111.
    else:
        return calorad


def emRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh):
    layerList=[0,1,2,4,5,6]
    return caloRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh,layerList)

def caloEffRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh,layerList):
    from math import cosh, fabs, fmod, sqrt
    # Calculate number of effective cells    
    numberCells = nEffCells(cellE)
    mycellE = list(cellE)
    tmpCellE=sorted(cellE,reverse=True)
#
    calorad = 0.
    sumET=0.
    pi=3.141592653589793 
    for icell in xrange(int(round(numberCells))):
        # hiIndex is the index in the original eneregy, eta, phi 
        # lists which correspond to the sorted highest E cells
        hiIndex=mycellE.index(tmpCellE[icell])
        deltaEta = fabs(jetEta - cellEta[hiIndex])
        dPhi = jetPhi - cellPhi[hiIndex]
        deltaPhi = fmod( dPhi+3*pi,2*pi )-pi        
        dR = sqrt( deltaEta*deltaEta + deltaPhi*deltaPhi )

        cellEnergy = cellE[hiIndex]
        cellET = cellE[hiIndex]/cosh(cellEta[hiIndex])
        cellSamp = cellSampling[hiIndex]
        if cellSamp in layerList:
            # include energy threshold possibility/cell
            if dR<0.4 and cellET>eThresh:
                calorad = calorad + dR*cellET
                sumET = sumET + cellET  
        # end cell loop
    if fabs(sumET)>0.000001:
        calorad = calorad/sumET
    return calorad

def emEffRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh):
    layerList=[0,1,2,4,5,6]
    return caloEffRadius(cellEta,cellPhi,cellE,cellSampling,jetEta,jetPhi,eThresh,layerList)
