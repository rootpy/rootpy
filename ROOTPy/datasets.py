import glob
import namedtuple
import os
import sys
import re
from xml.dom import minidom

mcpattern = re.compile("^group(?P<year>[0-9]+).perf-tau.mc(?P<prodyear>[0-9]+)_(?P<energy>[0-9]+)TeV.(?P<run>[0-9]+).(?P<name>).(?P<tag>[^.]+).(?P<suffix>.+)$")
datapattern = re.compile("^group(?P<year>[0-9]+).(?P<group>[^.]+).(?P<run>[0-9]+).(?P<stream>[^.]+).(?P<tag>[^.]+).(?P<version>[0-9\-]+).D3PD(?:.(?P<edition>[0-9]+))?_StreamD3PD_Tau(?P<size>SMALL$|MEDIUM$)")

Dataset = namedtuple.namedtuple( 'Dataset', 'name datatype classtype treename weight files' )

classes = {
    'BACKGROUND' :0,
    'SIGNAL'     :1
}

types = {
    'DATA' :0,
    'MC'   :1
}

data_periods = {
    'AB' :xrange(152166,155161),
    'C'  :xrange(155228,156683),
    'D'  :xrange(158045,159225),
    'E'  :xrange(160387,161949),
    'F'  :xrange(162347,162883),
    'G'  :xrange(165591,166384),
    'H'  :xrange(166466,166965)
}

if not os.environ.has_key('DATAROOT'):
    sys.exit("DATAROOT not defined!")
dataroot = os.environ['DATAROOT']

def get_sample(name, periods=None):

    base = os.path.join(dataroot,name)
    if not os.path.isdir(base):
        print "Sample %s not found at %s"%(name,base)
        return None
    metafile = os.path.join(base,'meta.xml')
    if not os.path.isfile(metafile):
        print "Metadata %s not found!"%metafile
        return None
    try:
        metafile = open(metafile,'r')
        doc = minidom.parse(metafile)
        metafile.close()
        meta = doc.getElementsByTagName("meta")
        datatype = meta[0].getElementsByTagName("type")[0].childNodes[0].nodeValue.upper()
        classname = meta[0].getElementsByTagName("class")[0].childNodes[0].nodeValue.upper()
        weight = float(eval(str(meta[0].getElementsByTagName("weight")[0].childNodes[0].nodeValue)))
        treename = str(meta[0].getElementsByTagName("tree")[0].childNodes[0].nodeValue)
    except:
        print "Could not parse metadata!"
        return None 
    if not classes.has_key(classname):
        print "Class %s is not defined!"%classname
        if len(classes) > 0:
            print "Use one of these:"
            for key in classes.keys():
                print key
        else:
            print "No classes have been defined!"
        return None
    classtype = classes[classname]
    if not types.has_key(datatype):
        print "Datatype %s is not defined!"%datatype
        if len(types) > 0:
            print "Use one of these:"
            for key in types.keys():
                print key
        else:
            print "No datatypes have been defined!"
    datatype = types[datatype]
    dirs = glob.glob(os.path.join(base,'*'))
    actualdirs = []
    for dir in dirs:
        if os.path.isdir(dir):
            actualdirs.append(dir)
    files = []
    if datatype == types['DATA']:
        # check for duplicate runs and take last edition
        runs = {}
        for dir in actualdirs:
            datasetname = os.path.basename(dir)
            match = re.match(datapattern,datasetname)
            if not match:
                print "Warning: directory %s is not a valid dataset name!"%datasetname
            else:
                runnumber = int(match.group('run'))
                if periods != None:
                    isinperiod = False
                    for period in periods:
                        if not data_periods.has_key(period):
                            print "Period %s is not defined!"%period
                            return None
                        if runnumber in data_periods[period]:
                            isinperiod = True
                            break
                    if not isinperiod:
                        continue
                edition = 0
                if match.group('edition'):
                    edition = int(match.group('edition'))
                if runs.has_key(runnumber):
                    print "Warning: multiple editions of dataset %s exist!"%datasetname
                    if edition > runs[runnumber]['edition']:
                        runs[runnumber] = {'edition':edition, 'dir':dir}
                else:
                    runs[runnumber] = {'edition':edition, 'dir':dir}
        for key,value in runs.items():
            files += glob.glob(os.path.join(value['dir'],'*root*'))
    else:
        for dir in actualdirs:
            files += glob.glob(os.path.join(dir,'*root*'))
    samplename = name
    if periods:
        samplename += "_%s"%("".join(periods))
    return Dataset(samplename,datatype,classtype,treename,weight,files)
    
"""
dataset = {}

#_______________________________________________________
DatasetTuple = namedtuple.namedtuple( 'DatasetTuple', 'runnumber tag generator treename weight files' )

dataset["data"]=[DatasetTuple( 111, "Data", "HCP2010", "tauPerf", 1.0, glob.glob(os.path.join(os.environ["data"],"DPD/data","group10.perf-tau.*.L1Calo-DESD_MET.*.00-06-00-02*TauMEDIUM/*root*")))]
dataset["data_small"]=[DatasetTuple( 111, "Data_small", "HCP2010", "tauPerf", 1.0, glob.glob(os.path.join(os.environ["data"],"DPD/data/small","group10.perf-tau.*.L1Calo-DESD_MET.*.00-06-00-02*TauSMALL/*root*")))]

base = os.path.join(os.environ["data"],"DPD/mc")

dataset["Ztautau"] = [
DatasetTuple( 106052, "Ztautau", "PythiaZtautau", "tauPerfSmall", 1.0, glob.glob( os.path.join(base,"group10.perf-tau.mc09_7TeV.106052.PythiaZtautau.AOD.e468_s765_s767_r1302_r1306.00-06-00-03.D3PD.*StreamD3PD_TauSMALL/*root*")))
]

##### DW TUNE ####################
# cross section 7.7745E+06, 361989 
dataset["PythiaDW"] = [
DatasetTuple( 115859, "DWJ0", "PythiaDW", "tauPerfSmall", 21.47717, glob.glob( os.path.join(base,"group10.perf-tau.mc09_7TeV.115859.J0_pythia_DW.e570_s766_s767_r1303_tid150952_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 5.0477E+05, 332993
DatasetTuple( 115860, "DWJ1", "PythiaDW", "tauPerfSmall", 1.51585769 , glob.glob( os.path.join(base,"group10.perf-tau.mc09_7TeV.115860.J1_pythia_DW.e570_s766_s767_r1303_tid150951_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 2.9366E+04, 370993 
DatasetTuple( 115861, "DWJ2", "PythiaDW", "tauPerfSmall", 0.079155132, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115861.J2_pythia_DW.e570_s766_s767_r1303_tid150950_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 1.5600E+03, 392997
DatasetTuple( 115862, "DWJ3", "PythiaDW", "tauPerfSmall", 0.003969496, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115862.J3_pythia_DW.e570_s766_s767_r1303_tid150949_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 6.4517E+01, 397986 
DatasetTuple( 115863, "DWJ4", "PythiaDW", "tauPerfSmall", 0.000162109, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115863.J4_pythia_DW.e570_s766_s767_r1303_tid150948_00.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")))
]
############################################
##### PERUGIA 2010 TUNE ####################
dataset["Perugia2010"] = [
# cross section 7.7788E+06, 398244 
DatasetTuple( 115849, "PerugiaJ0", "PythiaP10", "tauPerfSmall", 19.5327, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115849.J0_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 5.0385E+05, 397942
DatasetTuple( 115850, "PerugiaJ1", "PythiaP10", "tauPerfSmall", 1.266139, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115850.J1_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 2.9353E+04, 397856
DatasetTuple( 115851, "PerugiaJ2", "PythiaP10", "tauPerfSmall", 0.07377795 , glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115851.J2_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 1.5608E+03, 398393 
DatasetTuple( 115852, "PerugiaJ3", "PythiaP10", "tauPerfSmall", 0.00391774 , glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115852.J3_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*"))),
# cross section 1.8760E+00, 397195
DatasetTuple( 115853, "PerugiaJ4", "PythiaP10", "tauPerfSmall", 0.000004723, glob.glob( os.path.join(base, "group10.perf-tau.mc09_7TeV.115853.J4_pythia_Perugia2010.e568_s766_s767_r1303.00-06-00-02.D3PD_StreamD3PD_TauSMALL/*root*")))
]

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
