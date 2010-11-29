import math

sign = lambda x: 1 if x>=0 else -1

def dphi(phi1, phi2):
    return abs(math.fmod(phi1 - phi2 + 3*math.pi ,2*math.pi) - math.pi)

def dR(eta1, phi1, eta2, phi2):
    _deta = abs( eta1 - eta2 )
    _dphi = dphi( phi1, phi2 )
    return math.sqrt(_deta*_deta + _dphi*_dphi)

def et_to_pt(et, eta, m):
    cosh_eta = math.cosh(eta)
    return math.sqrt( et*et - (m*m)/(cosh_eta*cosh_eta) )

def pt_to_et(pt, eta, m):
    cosh_eta = math.cosh(eta)
    return math.sqrt( pt*pt + (m*m)/(cosh_eta*cosh_eta) )
