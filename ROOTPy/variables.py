from ROOTPy.types import *

# Default value for tau variables (outside of any possible range)
DEFAULT = -9999.0
variables = {}

LLH = [
("author","author","tau_author","I"),
("eta","eta","tau_eta","F"),
("numTrack","numTrack","tau_numTrack","I"),
("et","et","tau_Et","F"),
("nPi0","nPi0","tau_nPi0","I"),
("emRadius","emRadius","tau_seedCalo_EMRadius","F"),
("emFractionCalib","EMFractionCalib","tau_calcVars_emFracCalib","F"),
("etOverPtLeadTrk","EToverpTLeadTrk","tau_etOverPtLeadTrk","F"),
("massTrkSys","massTrkSys","tau_massTrkSys","F"),
("trkAvgDist","trkAvgDist","tau_seedCalo_trkAvgDist","F"),
("topoInvMass","topoInvMass","tau_calcVars_topoInvMass","F")
]


variables["binning"] = {
    "author"  : {"var":Int(DEFAULT),   "range":(0,2)},
    "numTrack": {"var":Int(DEFAULT),   "range":(0,4)},
    "eta"     : {"var":Float(DEFAULT), "range":(-2.5,2.5)},
    "phi"     : {"var":Float(DEFAULT), "range":(-3.14159,3.14159)},
    "e"       : {"var":Float(DEFAULT), "range":(0.0,100.0)},
    "et"      : {"var":Float(DEFAULT), "range":(0.0,100.0)},
    "et_EMJES"      : {"var":Float(DEFAULT), "range":(0.0,100.0)},
    "pt"      : {"var":Float(DEFAULT), "range":(0.0,100.0)},
    "m"       : {"var":Float(DEFAULT), "range":(0.0,100.0)}
}

# fix eteflowoveret for next dump
variables["variables"] = {
    "charge"                     : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(-4,4),       "error":(0.,0.,"fixed")},
    "emRadius"                   : {"var":Float(DEFAULT), "safe":1, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.3),    "error":(0.,0.,"fixed")},
    "hadRadius"                  : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.3),    "error":(0.,0.,"fixed")},
    "isolFrac"                   : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.0),    "error":(0.,0.,"fixed")},
    "centFrac"                   : {"var":Float(DEFAULT), "safe":0, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.0),    "error":(0.,0.,"fixed")},
    "stripWidth2"                : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.05),   "error":(0.,0.,"fixed")},
    "nStrip"                     : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    "trFlightPathSig"            : {"var":Float(DEFAULT), "safe":0, "enable":1, "authors":[1,3], "prongs":[3],   "range":(-10.0,10.0), "error":(0.,0.,"fixed")},
    "ipSigLeadTrk"               : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(-4.0,4.0),   "error":(0.,0.,"fixed")},
    "ipSigLeadLooseTrk"          : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(-4.0,4.0),   "error":(0.,0.,"fixed")},
    "nIsolLooseTrk"              : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,6),        "error":(0.,0.,"fixed")},
    "EToverpTLeadTrk"            : {"var":Float(DEFAULT), "safe":1, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,6.0),    "error":(0.,0.,"fixed")},
    "EToverpTLeadTrk_EMJES"      : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,6.0),    "error":(0.,0.,"fixed")},
    #"pTLeadTrkoverET"            : {"var":Float(DEFAULT), "safe":1, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.2),    "error":(0.,0.,"fixed")},
    "ipz0SinThetaSigLeadTrk"     : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(-4.0,4.0),   "error":(0.,0.,"fixed")},
    "massTrkSys"                 : {"var":Float(DEFAULT), "safe":0, "enable":1, "authors":[1,3], "prongs":[3],   "range":(0.0,3.0),    "error":(0.,0.,"fixed")},
    "trkWidth2"                  : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[3],   "range":(0.0,0.001),  "error":(0.,0.,"fixed")},
    "nPi0"                       : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[3],   "prongs":[1,3], "range":(0,7),        "error":(0.,0.,"fixed")},
    "nIsolTrk"                   : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[3],   "prongs":[1,3], "range":(0,4),        "error":(0.,0.,"fixed")},
    "ETeflow_over_ET"            : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[3],   "prongs":[1,3], "range":(0.3,1.8),    "error":(0.,0.,"fixed")},
    "trkAvgDist"                 : {"var":Float(DEFAULT), "safe":1, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.3),    "error":(0.,0.,"fixed")},
    #"ETHadAtEMScale"             : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETEMAtEMScale"              : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"ETHadCalib"                 : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"ETEMCalib"                  : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"sumpT3Trk"                  : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"sumpT"                      : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"dRmin"                      : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.4),    "error":(0.,0.,"fixed")},
    #"dRmax"                      : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.4),    "error":(0.,0.,"fixed")},
    #"sumpT_over_ET"              : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"sumpT3trk_over_ET"          : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETHad_EM_over_sumpT3Trk"    : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETEM_EM_over_sumpT3Trk"     : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"ETHad_Calib_over_sumpT3Trk" : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETEM_Calib_over_sumpT3Trk"  : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"ETHad_EM_over_sumpT"        : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETEM_EM_over_sumpT"         : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    #"ETHad_Calib_over_sumpT"     : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.5),    "error":(0.,0.,"fixed")},
    #"ETEM_Calib_over_sumpT"      : {"var":Float(DEFAULT), "safe":1, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,4.0),    "error":(0.,0.,"fixed")},
    "EMFractionCalib"            : {"var":Float(DEFAULT), "safe":0, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.2),    "error":(0.,0.,"fixed")}, 
    "EMFractionCalib_EMJES"      : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.2),    "error":(0.,0.,"fixed")}, 
    "EMFractionAtEMScale"        : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.2),    "error":(0.,0.,"fixed")},
    "topoInvMass"                : {"var":Float(DEFAULT), "safe":0, "enable":1, "authors":[1,3], "prongs":[1,3], "range":(0.0,8.0),    "error":(0.,0.,"fixed")},
    "topoInvMass_recalc"         : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,8.0),    "error":(0.,0.,"fixed")},
    "topoInvMass_EMJES"          : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,8.0),    "error":(0.,0.,"fixed")},
    "effTopoInvMass"             : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,8.0),    "error":(0.,0.,"fixed")},
    "numTopoClusters"            : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    "numEffTopoClusters"         : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,15.),    "error":(0.,0.,"fixed")},
    "topoMeandR"                 : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.4),    "error":(0.,0.,"fixed")},
    "effTopoMeandR"              : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,0.4),    "error":(0.,0.,"fixed")},
    #"NTRTHTHITSLEADTRK"          : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    #"NTRTHTOUTLIERSLEADTRK"      : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    #"NTRTHITSLEADTRK"            : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    #"NTRTOUTLIERSLEADTRK"        : {"var":Int(DEFAULT),   "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0,30),       "error":(0.,0.,"fixed")},
    "TRT_NHT_over_NLT"           : {"var":Float(DEFAULT), "safe":0, "enable":0, "authors":[1,3], "prongs":[1,3], "range":(0.0,1.0),    "error":(0.,0.,"fixed")}
}

fancy = {
    "BDTJetScore"       : {"fancy":"BDT", "units":"", "scale":""},
    "llhsafe3"          : {"fancy":"Likelihood", "units":"", "scale":""},
    "LLH"               : {"fancy":"Likelihood", "units":"", "scale":""},
    "likelihood"        : {"fancy":"Likelihood", "units":"", "scale":""},
    "eta"               : {"fancy":"#font[152]{#eta}", "units":"", "scale":""},
    "phi"               : {"fancy":"#font[152]{#phi}", "units":"", "scale":""},
    "e"                 : {"fancy":"#font[52]{E}", "units":"GeV", "scale":"/1000"},
    "eVisOfMatch"       : {"fancy":"#font[52]{E}^{vis} (of match)", "units":"GeV", "scale":"/1000"},
    "et"                : {"fancy":"#font[52]{E}_{T}", "units":"GeV", "scale":"/1000"},
    "et_EMJES"          : {"fancy":"#font[52]{E}_{T}^{EM+JES}", "units":"GeV", "scale":"/1000"},
    "etVisOfMatch"      : {"fancy":"#font[52]{E}_{T}^{vis} (of match)", "units":"GeV", "scale":"/1000"},
    "pt"                : {"fancy":"#font[52]{p}_{T}", "units":"GeV", "scale":"/1000"},
    "numTrack"          : {"fancy":"#Tracks", "units":"", "scale":""},
    "emRadius"          : {"fancy":"#font[52]{R}_{EM}", "units":"", "scale":""},
    "hadRadius"         : {"fancy":"#font[52]{R}_{Had}", "units":"", "scale":""},
    "isolFrac"          : {"fancy":"#font[52]{F}_{ring}", "units":"", "scale":""},
    "centFrac"          : {"fancy":"#font[52]{F}_{core}", "units":"", "scale":""}, 
    "EToverpTLeadTrk"   : {"fancy":"#font[52]{E}_{T}/#font[52]{p}_{1T}","units":"", "scale":""},
    "EToverpTLeadTrk_EMJES"   : {"fancy":"#font[52]{E}_{T}^{EM+JES}/#font[52]{p}_{1T}","units":"", "scale":""},
    "massTrkSys"        : {"fancy":"#font[52]{M}_{trk}","units":"GeV", "scale":"/1000"},
    "trkAvgDist"        : {"fancy":"<#Delta#font[52]{R}_{trk}>","units":"", "scale":""},
    "EMFractionCalib"   : {"fancy":"#font[52]{F}_{EM}","units":"", "scale":""},
    "EMFractionCalib_EMJES"   : {"fancy":"#font[52]{F}_{EM}^{EM+JES}","units":"", "scale":""},
    "topoInvMass"       : {"fancy":"#font[52]{M}_{topo}","units":"GeV", "scale":"/1000"},
    "topoInvMass_recalc" : {"fancy":"#font[52]{M}_{topo}^{EM+JES}","units":"GeV", "scale":"/1000"},
    "topoInvMass_EMJES" : {"fancy":"#font[52]{M}_{topo}^{EM+JES}","units":"GeV", "scale":"/1000"},
    "effTopoInvMass"    : {"fancy":"#font[52]{M}_{topo}^{eff}","units":"GeV", "scale":"/1000"},
    "TRT_NHT_over_NLT"  : {"fancy":"#font[52]{High Threshold / Low Threshold TRT Hits}","units":"", "scale":""},
    "nPi0"              : {"fancy":"##font[152]{#pi}^{0}","units":"", "scale":""}
}

TD2DPD = {
#    "BDT":"tau_BDTJetScore",
    "BDTJetScore":"tau_BDTJetScore",
    "LLH":"tau_likelihood",
    "likelihood":"tau_likelihood",
    "llhsafe":"llhsafe3",
    "author"  : "tau_author",
    "numTrack": "tau_numTrack",
    "eta"     : "tau_eta",
    "phi"     : "tau_phi",
    "et"      : "tau_Et",
    "et_EMJES": "tau_Et_EMJES",
    "pt"      : "tau_pt",
    "charge"                              : "tau_charge",
    "emRadius"                            : "tau_seedCalo_EMRadius",
    "hadRadius"                           : "tau_seedCalo_hadRadius",
    "isolFrac"                            : "tau_seedCalo_isolFrac",
    "centFrac"                            : "tau_seedCalo_centFrac",
    "stripWidth2"                         : "tau_seedCalo_stripWidth2",
    "numStripCells"                       : "tau_seedCalo_nStrip",
    "trFlightPathSig"                     : "tau_trFlightPathSig",
    "ipSigLeadTrack"                      : "tau_ipSigLeadTrack",
    "EToverpTLeadTrk"                     : "tau_etOverPtLeadTrk",
    "EToverpTLeadTrk_EMJES"               : "tau_etOverPtLeadTrk_EMJES",
    "z0SinThetaSig"                       : "tau_ipZ0SinThetaSigLeadTrk",
    "massTrkSys"                          : "tau_massTrkSys",
    "trkWidth2"                           : "tau_trkWidth2",
    "nPi0"                                : "tau_nPi0",
    "nAssocTracksIsol"                    : "tau_seedTrk_nIsolTrk",
    "trkAvgDist"                          : "tau_seedCalo_trkAvgDist", 
    "etHad_EMScale_over_SumPT_max3tracks" : "tau_calcVars_etHadSumPtTracks",
    "etEM_EMScale_over_SumPT_max3tracks"  : "tau_calcVars_etEMSumPtTracks",
    "topoInvMass"                         : "tau_calcVars_topoInvMass",
    "topoInvMass_recalc"                  : "tau_calcVars_topoInvMass_recalc",
    "topoInvMass_EMJES"                   : "tau_calcVars_topoInvMass_EMJES",
    "effTopoInvMass"                      : "tau_calcVars_effTopoInvMass",
    "numTopoClusters"                     : "tau_calcVars_numTopoClusters",
    "numEffTopoClusters"                  : "tau_calcVars_numEffTopoClusters",
    "topoMeandR"                          : "tau_calcVars_topoMeanDeltaR",
    "effTopoMeandR"                       : "tau_calcVars_effTopoMeanDeltaR",
    "EMFractionCalib"                     : "tau_calcVars_emFracCalib",
    "EMFractionCalib_EMJES"               : "tau_calcVars_emFracCalib_EMJES",
}

DPD2TD = dict([(value,key) for key,value in TD2DPD.items()])

topoClusterVariables = ["effTopoMeandR", "topoMeandR", "numEffTopoClusters", "numTopoClusters", "effTopoInvMass", "topoInvMass"]

variables["flags"] = {
    "index"                     : {"var":Int(DEFAULT),   "range":(0,1)},
    "run"                       : {"var":Int(DEFAULT),   "range":(0,0)},
    "lumiblock"                 : {"var":Int(DEFAULT),   "range":(0,0)},
    "event"                     : {"var":Int(DEFAULT),   "range":(0,0)}
}

variables["RecoJet"] = {
    "matchesTruth"              : {"var":Int(DEFAULT),   "range":(-1,1)},
    "pdgid"                     : {"var":Int(DEFAULT),   "range":(-1,21)},
    "numProngsOfMatch"          : {"var":Int(DEFAULT),   "range":(0,3)},
    "eVisOfMatch"               : {"var":Float(DEFAULT), "range":(0.,100.)},
    "etVisOfMatch"              : {"var":Float(DEFAULT), "range":(0.,100.)},
    "ptVisOfMatch"              : {"var":Float(DEFAULT), "range":(0.,100.)},
    "etaVisOfMatch"             : {"var":Float(DEFAULT), "range":(-3.,3.)}
}

variables["TruthTau"] = {
    "matchesReco"               : {"var":Int(DEFAULT),   "range":(-1,1)},
    "numTracksOfMatch"          : {"var":Int(DEFAULT),   "range":(0,8)},
    "numProng"                  : {"var":Int(DEFAULT),   "range":(0,3)},
    "eVis"                      : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "etVis"                     : {"var":Float(DEFAULT), "range":(0.,100000.)}, 
    "ptVis"                     : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "etaVis"                    : {"var":Float(DEFAULT),"range":(-3,3)},
    "eOfMatch"                  : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "etOfMatch"                 : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "ptOfMatch"                 : {"var":Float(DEFAULT), "range":(0.,100000.)}
}

variables["TruthJet"] = {
    "etaVis"                    : {"var":Float(DEFAULT),"range":(-3.,3.)},
    "matchesReco"               : {"var":Int(DEFAULT),   "range":(-1,1)},
    "pdgid"                     : {"var":Int(DEFAULT),   "range":(0,21)},
    "eVis"                      : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "etVis"                     : {"var":Float(DEFAULT), "range":(0.,100000.)}, 
    "ptVis"                     : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "eOfMatch"                  : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "etOfMatch"                 : {"var":Float(DEFAULT), "range":(0.,100000.)},
    "ptOfMatch"                 : {"var":Float(DEFAULT), "range":(0.,100000.)}
}

variables["discriminants"] = {
    "likelihood"                : {"var":Float(DEFAULT), "range":(-50.0,50.0)},
    #"llhsafe"                   : {"var":Float(DEFAULT), "range":(-50.0,50.0)},
    "BDTJetScore"               : {"var":Float(DEFAULT), "range":(0.0,1.0)},
    "ElectronVetoLoose"         : {"var":Float(DEFAULT), "range":(0.0,1.0)},
    "ElectronVetoMedium"        : {"var":Float(DEFAULT), "range":(0.0,1.0)},
    "ElectronVetoTight"         : {"var":Float(DEFAULT), "range":(0.0,1.0)}
}

def intersect(a, b):
     return list(set(a) & set(b))

def getVariables(authors,prongs,safe=False,topoClusterOnly=False,enabled=False):
    
    map = {}
    for key,value in variables["variables"].items():
        if (enabled and not value["enable"]) and not safe and not topoClusterOnly:
            continue
        if safe and not value["safe"]:
            continue
        if topoClusterOnly and key not in topoClusterVariables:
            continue
        if len(intersect(value["prongs"],prongs))==0:
            continue
        if len(intersect(value["authors"],authors))==0:
            continue
        map[key] = value
    return map

variablesCaps = dict([(var.upper(),var) for var in variables["variables"].keys()+variables["binning"].keys()])

def getRange(variable):
    
    for key in variables.keys():
        if variable in variables[key].keys():
            return variables[key][variable]["range"]
    return None

def getVariable(variable):
    
    for key in variables.keys():
        if variable in variables[key].keys():
            return variables[key][variable]["var"]
    return None

def getRecoVariableNames():
    
    return binningVariables.keys()+variables.keys()+flags.keys()

def getMapFromList(varList):
    
    map = {}
    for variable in varList:
        tmpVar = variablesCaps[variable.upper().strip()]
        map[tmpVar] = variables["variables"][tmpVar]
    return map

def getBinningVariables():
    
    map = {}
    for variable in variables["binning"].keys():
        map[variable] = variables["binning"][variable]["var"]
    return map

def getVarMap(types):
    
    map = {}
    for type in types:
        if type in variables.keys():
            for key,value in variables[type].items():
                map[key] = value["var"]
        else:
            print "variables of type %s do not exist"%type
    return map

def getRangeMap(types):
    
    map = {}
    for type in types:
        if type in variables.keys():
            for key,value in variables[type].items():
                map[key] = {"min":value["range"][0],"max":value["range"][1],"type":value["var"].type()}
        else:
            print "variables of type %s do not exist"%type
    return map
