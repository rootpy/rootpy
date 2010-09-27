import ROOT, glob, namedtuple
import os
import copy

dataset = {}

#_______________________________________________________
DatasetTuple = namedtuple.namedtuple( 'DatasetTuple', 'runnumber tag generator weight files' )

newfiles = glob.glob(os.environ["data"]+"DPD/data"+"group10.perf-tau.*.L1Calo-DESD_MET.*.00-06-00-02*TauMEDIUM/*root*")
nfiles = copy.copy(newfiles)
for file in nfiles:
    if file.startswith(base+"group10.perf-tau.159224"):
        newfiles.remove(file)

dataset["data"]=[DatasetTuple( 111, "Data", "HCP2010", 1.0, newfiles)]

base = os.environ["data"]+"DPD/mc"

##### DW TUNE ####################
# cross section 7.7745E+06, 361989 
dataset["PythiaDW"] = [
DatasetTuple( 115859, "DWJ0", "PythiaDW", 21.47717, glob.glob( base + "group10.perf-tau.mc09_7TeV.115859.J0_pythia_DW.e570_s766_s767_r1303_tid150952_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 5.0477E+05, 332993
DatasetTuple( 115860, "DWJ1", "PythiaDW",1.51585769 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115860.J1_pythia_DW.e570_s766_s767_r1303_tid150951_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 2.9366E+04, 370993 
DatasetTuple( 115861, "DWJ2", "PythiaDW", 0.079155132, glob.glob( base + "group10.perf-tau.mc09_7TeV.115861.J2_pythia_DW.e570_s766_s767_r1303_tid150950_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 1.5600E+03, 392997
DatasetTuple( 115862, "DWJ3", "PythiaDW", 0.003969496, glob.glob( base + "group10.perf-tau.mc09_7TeV.115862.J3_pythia_DW.e570_s766_s767_r1303_tid150949_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 6.4517E+01, 397986 
DatasetTuple( 115863, "DWJ4", "PythiaDW",0.000162109, glob.glob( base + "group10.perf-tau.mc09_7TeV.115863.J4_pythia_DW.e570_s766_s767_r1303_tid150948_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
]
############################################
##### PERUGIA 2010 TUNE ####################
dataset["Perugia2010"] = [
# cross section 7.7788E+06, 398244 
DatasetTuple( 115849, "PerugiaJ0", "PythiaP10",19.5327, glob.glob( base + "group10.perf-tau.mc09_7TeV.115849.J0_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 5.0385E+05, 397942
DatasetTuple( 115850, "PerugiaJ1", "PythiaP10", 1.266139, glob.glob( base + "group10.perf-tau.mc09_7TeV.115850.J1_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 2.9353E+04, 397856
DatasetTuple( 115851, "PerugiaJ2", "PythiaP10",0.07377795 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115851.J2_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 1.5608E+03, 398393 
DatasetTuple( 115852, "PerugiaJ3", "PythiaP10",0.00391774 , glob.glob( base + "group10.perf-tau.mc09_7TeV.115852.J3_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")),
# cross section 1.8760E+00, 397195
DatasetTuple( 115853, "PerugiaJ4", "PythiaP10", 0.000004723, glob.glob( base + "group10.perf-tau.mc09_7TeV.115853.J4_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))
]

"""
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
"""
