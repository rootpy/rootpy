sampleGroupsDict = {"J0-J4"        : {"idlist":[105009,105010,105011,105012,105013],"class":0,"name":"J0-J4"},
                    "J0-J5"        : {"idlist":[105009,105010,105011,105012,105013,105014],"class":0,"name":"J0-J5"},
                    "J1-J3"        : {"idlist":[105010,105011,105012],"class":0,"name":"J1-J3"},
                    "J1-J4"        : {"idlist":[105010,105011,105012,105013],"class":0,"name":"J1-J4"},
                    "J1-J5"        : {"idlist":[105010,105011,105012,105013,105014],"class":0,"name":"J1-J5"},
                    "J2-J5"        : {"idlist":[105011,105012,105013,105014],"class":0,"name":"J2-J5"},
                    "J0"           : {"idlist":[105009],"class":0,"name":"J0"},
                    "J1"           : {"idlist":[105010],"class":0,"name":"J1"},
                    "J2"           : {"idlist":[105011],"class":0,"name":"J2"},
                    "J3"           : {"idlist":[105012],"class":0,"name":"J3"},
                    "J4"           : {"idlist":[105013],"class":0,"name":"J4"},
                    "J5"           : {"idlist":[105014],"class":0,"name":"J5"},
                    "JX_n5"        : {"idlist":[105010.5,105011.5,105012.5],"class":0,"name":"J1-J3 with pileup (n=5)"},
                    "WmunuNp0"     : {"idlist":[107690],"class":0},
                    "WmunuNp1"     : {"idlist":[107691],"class":0},
                    "WmunuNp2"     : {"idlist":[107692],"class":0},
                    "WmunuNp3"     : {"idlist":[107693],"class":0},
                    "WmunuNp4"     : {"idlist":[107694],"class":0},
                    "WmunuNp5"     : {"idlist":[107695],"class":0},
                    "Wmunu"        : {"idlist":[107690,107691,107692,107693,107694,107695],"class":0},
                    "semilepttbar" : {"idlist":[105200],"class":1,"name":"Semileptonic t#bar{t}"},
                    "hadttbar"     : {"idlist":[105204],"class":0,"name":"Hadronic t#bar{t}"},
                    "Ztautau"      : {"idlist":[106052],"class":1,"name":"Pythia Z#rightarrow#tau#tau"},
                    "Zee"          : {"idlist":[106046],"class":0,"name":"Z#rightarrowee"},
                    "Wenu"         : {"idlist":[106043],"class":0,"name":"W#rightarrowe#nu"},
                    "Atautau"      : {"idlist":[109870,109871,109874],"class":1,"name":"A#rightarrow#tau#tau"},
                    "data"         : {"idlist":[105000],"class":0,"name":"Data"},
                    "Ztautau"      : {"idlist":[106052],"class":1,"name":"Pythia Z#rightarrow#tau#tau"},
                    "Ztautau_n5"   : {"idlist":[106052.5],"class":1,"name":"Pythia Z#rightarrow#tau#tau (<n>=5)"},
                    "Ztautau_n2"   : {"idlist":[106052.2],"class":1,"name":"Pythia Z#rightarrow#tau#tau (<n>=2)"},
                    "Wtaunu"       : {"idlist":[107054],"class":1,"name":"Pythia Z#rightarrow#tau#tau"},
                    "Wtaunu_n2"    : {"idlist":[107054.2],"class":1,"name":"Pythia Z#rightarrow#tau#tau (<n>=2)"},
                    "Wtaunu_n5"    : {"idlist":[107054.5],"class":1,"name":"Pythia Z#rightarrow#tau#tau (<n>=5)"}}


def getName(sample):

    return ", ".join([sampleGroupsDict[s]["name"] for s in sample.split("+")])
    
"""    
    uniqueFinalStates = []
    for s in subsamples:
        if "#rightarrow" in s:
            finalState = s.split("#rightarrow")[1]
            if finalState not in uniqueFinalStates:
                uniqueFinalStates.append(finalState)
    sampleName = ""
    for s in subsamples:
"""
""" dictionary with all 7TeV cross-sections (in nb) per sample: 'xsec' = sigma x eff."""
xSectionDict7TeV = {
                     105000:{"name":"data","xsec":-1.},
                     115000:{"name":"PythiaZtautau","xsec":-1.},
                     115000.5:{"name":"DWZtautau","xsec":-1.},
                     125000:{"name":"DWJ1","xsec":-1.},
                     135000:{"name":"DWJ2","xsec":-1.},
                     145000:{"name":"DWJ3","xsec":-1.},
                     155000:{"name":"DWJ4","xsec":-1.},
                     165000:{"name":"PerugiaJ1","xsec":-1.},
                     175000:{"name":"PerugiaJ2","xsec":-1.},
                     185000:{"name":"PerugiaJ3","xsec":-1.},
                     195000:{"name":"PerugiaJ4","xsec":-1.},
                     265000:{"name":"materialJ1","xsec":-1.},
                     275000:{"name":"materialJ2","xsec":-1.},
                     285000:{"name":"materialJ3","xsec":-1.},
                     295000:{"name":"materialJ4","xsec":-1.},
                     105001:{"name":"pythia_minbias","xsec":-1.},
                     105009:{"name":"J0_pythia_jetjet","xsec":9.8534E+06},
                     105010:{"name":"J1_pythia_jetjet","xsec":6.7803E+05},
                     105011:{"name":"J2_pythia_jetjet","xsec":4.0979E+04},
                     105012:{"name":"J3_pythia_jetjet","xsec":2.1960E+03},
                     105013:{"name":"J4_pythia_jetjet","xsec":8.7701E+01},
                     105014:{"name":"J5_pythia_jetjet","xsec":2.3483E+00},
                     105010.5:{"name":"J1_pythia_jetjet_n5","xsec":6.7803E+05},
                     105011.5:{"name":"J2_pythia_jetjet_n5","xsec":4.0979E+04},
                     105012.5:{"name":"J3_pythia_jetjet_n5","xsec":2.1960E+03},
                     106052:{"name":"PythiaZtautau","xsec":8.5402E-01},
                     106052.5:{"name":"PythiaZtautau_n5","xsec":8.5402E-01},
                     106052.2:{"name":"PythiaZtautau_n2","xsec":8.5402E-01},
                     106046:{"name":"PythiaZee","xsec":8.5575E-01},
                     106043:{"name":"PythiaWenu","xsec":8.9380E+00},
                     107054:{"name":"PythiaWhadtaunu","xsec":8.9295E+00},
                     107054.2:{"name":"PythiaWhadtaunu_n2","xsec":8.9295E+00},
                     107054.5:{"name":"PythiaWhadtaunu_n5","xsec":8.9295E+00},
                     106573:{"name":"PythiabbAtautauMA800TB35","xsec":1.6648E-06},
                     109870:{"name":"PythiaAtautauMA120TB20_n2","xsec":.0230548},
                     109871:{"name":"PythiaAtautauMA300TB20_n2","xsec":.000406063},
                     109872:{"name":"PythiaAtautauMA100TB20","xsec":.0489015},
                     109873:{"name":"PythiaAtautauMA150TB20","xsec":.00907153},
                     109874:{"name":"PythiaAtautauMA200TB20_n2","xsec":.00259947},
                     109875:{"name":"PythiaAtautauMA110TB20","xsec":.0331659},
                     109876:{"name":"PythiaAtautauMA130TB20","xsec":.0162838},
                     109877:{"name":"PythiaAtautauMA140TB20","xsec":.0121798},
                     105200:{"name":"T1_McAtNlo_Jimmy","xsec":1.4412E-01},
                     105204:{"name":"TTbar_FullHad_McAtNlo_Jimmy","xsec":1.4428E-01}
                   }
