import math

sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0

dphi = lambda phi1, phi2 : abs(math.fmod((math.fmod(phi1, 2*math.pi) - math.fmod(phi2, 2*math.pi)) + 3*math.pi, 2*math.pi) - math.pi)

dR = lambda eta1, phi1, eta2, phi2: math.sqrt((eta1 - eta2)**2 + dphi(phi1, phi2)**2)

et_to_pt = lambda et, eta, m: math.sqrt(et**2 - (m**2)/(math.cosh(eta)**2))

pt_to_et = lambda pt, eta, m: math.sqrt(pt**2 + (m**2)/(math.cosh(eta)**2))
