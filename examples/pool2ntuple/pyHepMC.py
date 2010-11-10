import ROOT
import PDG
import sys

def log(message):
    
    print "HepMC: "+message

def getGenParticles(genEvent):
    
    priVertex = genEvent.barcode_to_vertex(-1)
    partons = getPartons(priVertex)
    partonJets = [getPartonJet(parton) for parton in partons]
    truthParticles = ROOT.DataVector("TruthParticle")()
    for parton in partonJets:
        if parton:
            tp = ROOT.TruthParticle(parton)
            truthParticles.push_back(tp)
            hold_implicit(truthParticles,tp)
        else:
            log("Found NULL parton: skipping!")
    return truthParticles

def hold_implicit(obj,member):

    if not hasattr(obj, '_implicit_members'):
        obj._implicit_members = []
    obj._implicit_members.append(member)

def getMCJets(genEvent):
    
    jets = []
    iter = genEvent.particles_begin()
    while iter != genEvent.particles_end():
        genParticle = iter.__deref__()
        if (genParticle.status() in range(141,145)) and (abs(genParticle.pdg_id()) in range(1,6)+[21]):
            jets.append(genParticle)
        iter.__preinc__()
    return jets

def getPartons(genVertex):
    
    partons = []
    iter = genVertex.particles_out_const_begin()
    while iter != genVertex.particles_out_const_end():
        particle = iter.__deref__()
        if abs(particle.pdg_id()) in range(1,6)+[21]:
            partons.append(particle)
        elif abs(particle.pdg_id()) == 6:
            childParticles = []
            log("top")
            bottoms = getChildren(particle,[5])
            log("   %i bottoms"%len(bottoms))
            childParticles += bottoms
            Wbosons = getChildren(particle,[24])
            log("   %i W bosons"%len(Wbosons))
            for W in Wbosons:
                lightquarks = getChildren(W,range(1,7))
                log("      %i light quarks"%len(lightquarks))
                childParticles += lightquarks
            partons+=childParticles
        elif particle.pdg_id() == 0:
            partons+=getChildren(particle,range(1,7)+[21])
        iter.__preinc__()
    return partons

def getChildren(particle,pdgids):
    
    children = []
    endVertex = particle.end_vertex()
    if endVertex:
        iter = endVertex.particles_out_const_begin()
        while iter != endVertex.particles_out_const_end():
            child = iter.__deref__()
            if abs(child.pdg_id()) in pdgids:
                children.append(child)
            if child.pdg_id() == particle.pdg_id():
                children += getChildren(child,pdgids)
            iter.__preinc__()
    return children

def getPartonJet(parton):
    
    if not parton:
        return None
    
    if parton.status() in range(141,145):
        return parton

    endVertex = parton.end_vertex()
    if endVertex:
        if endVertex.particles_out_size() != 1:
            log("error in finding parton jet: multiple outgoing children")
            log("adding parton with status of %i"%parton.status())
            return parton
        else:
            iter = endVertex.particles_out_const_begin()
            return getPartonJet(iter.__deref__())
    else:
        log("no jet found for parton")
        log("adding parton with status of %i"%parton.status())
        return parton
