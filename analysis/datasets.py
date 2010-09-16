import ROOT, glob, namedtuple

Data={}
MC={}
#_______________________________________________________
DatasetTuple = namedtuple.namedtuple( 'DatasetTuple', 'runnumber tag generator sigma files' )

host = "lhc03.phys.sfu.ca"
base = "/scratch1/oneil/data/"

Data["153565"]=DatasetTuple( 153565, "Data", "153565", 1.0, glob.glob( base + "group10.perf-tau.153565.L1Calo-DESD_MET.r1297_p157_p159.00-06-00-02.GRL.D3PD.1_StreamD3PD_TauSMALL/*root*"))

Data["HCP2010"]=DatasetTuple( 99999, "Data", "HCP2010", 1.0, glob.glob( base + "group10.perf-tau.*00-06-00-02*TauSMALL/*root*"))

Data["888888"]=DatasetTuple( 888888, "Data", "888888", 1.0, glob.glob( base + "group10.perf-tau.153565.00-06-00-TEST-3.D3PD_StreamD3PD_TauSMALL/*root*"))

newfiles = glob.glob(base+"group10.perf-tau.*.L1Calo-DESD_MET.*.00-06-00-02*TauMEDIUM/*root*")
for file in newfiles:
    if file.startswith(base+"group10.perf-tau.159224"):
        newfiles.remove(file)

#newfiles.remove("/scratch1/oneil/data/group10.perf-tau.158632.L1Calo-DESD_MET.f274_m544.00-06-00-02.GRL.D3PD_StreamD3PD_TauMEDIUM/group10.perf-tau.158632.L1Calo-DESD_MET.f274_m544.00-06-00-02.GRL.D3PD.StreamD3PD_TauMEDIUM._00018.root")
#newfiles.remove("/scratch1/oneil/data/group10.perf-tau.158632.L1Calo-DESD_MET.f274_m544.00-06-00-02.GRL.D3PD_StreamD3PD_TauMEDIUM/group10.perf-tau.158632.L1Calo-DESD_MET.f274_m544.00-06-00-02.GRL.D3PD.StreamD3PD_TauMEDIUM._00072.root")

Data["HCP2010_M"]=DatasetTuple( 111, "Data", "111", 1.0, newfiles)

##############################################
#Now Add MC
# Each weight is cross section/number of events
# These are needed to properly combine JX samples. 
# If it is not a JX sample, just set the weight to 1.
#
base = "/scratch1/oneil/MC/"



####
MC = { "105001" : DatasetTuple( 105001, "MinBias", "Pythia", 1.0, glob.glob( base + "group10.perf-tau.mc09_7TeV.105001.pythia_minbias.e517_s764_s767_r1302_tid136439_00.DESD_MET.00-05-03.tauPerfD3PD/*root*"))}

MC["105009"]=DatasetTuple( 105009, "J0", "Pythia", 9856800., glob.glob( base + "group10.perf-tau.mc09_7TeV.105009.J0_pythia_jetjet.recon.ESD.e468_s766_s767_r1303.00-05-03.tauPerfD3PD/*root*"))

MC["105010"]=DatasetTuple( 105010, "J1", "Pythia", 678080., glob.glob( base + "group10.perf-tau.mc09_7TeV.105010.J1_pythia_jetjet.recon.ESD.e468_s766_s767_r1303.00-05-03.tauPerfD3PD/*root*"))

MC["105011"]=DatasetTuple( 105011, "J2", "Pythia", 40994., glob.glob( base + "group10.perf-tau.mc09_7TeV.105011.J2_pythia_jetjet.recon.ESD.e468_s766_s767_r1303.00-05-03.tauPerfD3PD/*root*"))

MC["105012"]=DatasetTuple( 105012, "J3", "Pythia", 2193.6, glob.glob( base + "group10.perf-tau.mc09_7TeV.105012.J3_pythia_jetjet.recon.ESD.e468_s766_s767_r1303.00-05-03.tauPerfD3PD.1/*root*"))

MC["105013"]=DatasetTuple( 105013, "J4", "Pythia", 87.704, glob.glob( base + "group10.perf-tau.mc09_7TeV.105013.J4_pythia_jetjet.recon.ESD.e468_s766_s767_r1303.00-05-03.tauPerfD3PD/*root*"))

##### DW TUNE ####################
# cross section 7.7745E+06, 361989 
MC["115859"]=DatasetTuple( 115859, "DWJ0", "PythiaDW", 21.47717, glob.glob( base + "group10.perf-tau.mc09_7TeV.115859.J0_pythia_DW.e570_s766_s767_r1303_tid150952_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 5.0477E+05, 332993
MC["115860"]=DatasetTuple( 115860, "DWJ1", "PythiaDW",1.51585769 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115860.J1_pythia_DW.e570_s766_s767_r1303_tid150951_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 2.9366E+04, 370993 
MC["115861"]=DatasetTuple( 115861, "DWJ2", "PythiaDW", 0.079155132, glob.glob( base + "group10.perf-tau.mc09_7TeV.115861.J2_pythia_DW.e570_s766_s767_r1303_tid150950_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 1.5600E+03, 392997
MC["115862"]=DatasetTuple( 115862, "DWJ3", "PythiaDW", 0.003969496, glob.glob( base + "group10.perf-tau.mc09_7TeV.115862.J3_pythia_DW.e570_s766_s767_r1303_tid150949_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 6.4517E+01, 397986 
MC["115863"]=DatasetTuple( 115863, "DWJ4", "PythiaDW",0.000162109, glob.glob( base + "group10.perf-tau.mc09_7TeV.115863.J4_pythia_DW.e570_s766_s767_r1303_tid150948_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
############################################
##### PERUGIA 2010 TUNE ####################
# cross section 7.7788E+06, 398244 
MC["115849"]=DatasetTuple( 115849, "PerugiaJ0", "PythiaP10",19.5327, glob.glob( base + "group10.perf-tau.mc09_7TeV.115849.J0_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 5.0385E+05, 397942
MC["115850"]=DatasetTuple( 115850, "PerugiaJ1", "PythiaP10", 1.266139, glob.glob( base + "group10.perf-tau.mc09_7TeV.115850.J1_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 2.9353E+04, 397856
MC["115851"]=DatasetTuple( 115851, "PerugiaJ2", "PythiaP10",0.07377795 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115851.J2_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 1.5608E+03, 398393 
MC["115852"]=DatasetTuple( 115852, "PerugiaJ3", "PythiaP10",0.00391774 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115852.J3_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross section 1.8760E+00, 397195
MC["115853"]=DatasetTuple( 115853, "PerugiaJ4", "PythiaP10", 0.000004723, glob.glob( base + "group10.perf-tau.mc09_7TeV.115853.J4_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
############################################
##### material model ####################
# cross=6.7808E+05, 99947 events 
MC["105010M"]=DatasetTuple( 105010, "J1", "PythiaMat", 6.7843, glob.glob( base + "group10.perf-tau.mc09_7TeV.105010.J1_pythia.e468_s790_s791_r1304.00-06-00-03.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross=4.0994E+04, 99748 events 
MC["105011M"]=DatasetTuple( 105011, "J2", "PythiaMat", 0.410976, glob.glob( base + "group10.perf-tau.mc09_7TeV.105011.J2_pythia.e468_s790_s791_r1304.00-06-00-03.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross=2.1936E+03, 73900 
MC["105012M"]=DatasetTuple( 105012, "J3", "PythiaMat", 0.029683, glob.glob( base + "group10.perf-tau.mc09_7TeV.105012.J3_pythia.e468_s790_s791_r1304.00-06-00-03.D3PD_StreamD3PD_TauSMALL/*root*"))
# cross=8.7704E+01, 99549
MC["105013M"]=DatasetTuple( 105013, "J4", "PythiaMat",0.000881013, glob.glob( base + "group10.perf-tau.mc09_7TeV.105013.J4_pythia.e468_s790_s791_r1304.00-06-00-03.D3PD_StreamD3PD_TauSMALL/*root*"))


###########################
# we also have signal
MC["106023"]=DatasetTuple( 106023, "Wtau", "Pythia", 1.0, glob.glob( base + "group10.perf-tau.mc09_7TeV.106023.PythiaWhadtaunu.recon.ESD.e468_s765_s767_r1302.00-05-03.tauPerfD3PD/*root*"))

MC["106052"]=DatasetTuple( 106052, "Ztautau", "Pythia", 1.0, glob.glob( base + "group10.perf-tau.mc09_7TeV.106052.PythiaZtautau.e468_s765_s767_r1302.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))

MC["107413"]=DatasetTuple( 107413, "ZtautauDW", "PythiaDW", 1.0, glob.glob( base + "group10.perf-tau.mc09_7TeV.107413.PythiaZtautau_DW.e579_s766_s767_r1303_r1306.00-06-00-03.D3PD.100826092550_StreamD3PD_TauSMALL/*root*"))

#
Data7TeV = {}
MC7TeV = {}
for k,v in Data.items():
    Data7TeV[k] = v

for k,v in MC.items():
    MC7TeV[k] = v

#_______________________________________________________
def getNumEvents( dataset ):
    fChain = ROOT.TChain( "tauPerf" )
    for f in dataset.files:
        fChain.Add( f )
    return fChain.GetEntries()

#_______________________________________________________
def getWeight( dataset ):
    # if the dataset is unwighted, always return 1.0
    # otherwise, take cross section/#events
    if dataset.sigma==1.0:
        return dataset.sigma
    else:
        return (dataset.sigma) / getNumEvents( dataset )
