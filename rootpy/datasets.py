import glob
from collections import namedtuple
import os
import sys
import re
import yaml

mcpattern = re.compile("^group(?P<year>[0-9]+).perf-tau.mc(?P<prodyear>[0-9]+)_(?P<energy>[0-9]+)TeV.(?P<run>[0-9]+).(?P<name>).(?P<tag>[^.]+).(?P<suffix>.+)$")
datapattern = re.compile("^group(?P<year>[0-9]+).(?P<group>[^.]+).(?P<run>[0-9]+).(?P<stream>[^.]+).(?P<tag>[^.]+).(?P<version>[0-9\-]+)(?:.(?P<grl>GRL))?.D3PD(?:.(?P<edition>[0-9]+))?_StreamD3PD_Tau(?P<size>SMALL$|MEDIUM$)")

Dataset = namedtuple('Dataset', 'name label datatype classtype treename weight files')

DATA,MC = range(2)
BACKGROUND,SIGNAL = range(2)
TAU,MUON,ELEC,JET = range(4)

classes = {
    'BACKGROUND': BACKGROUND,
    'SIGNAL'    : SIGNAL
}

types = {
    'DATA': DATA,
    'MC'  : MC
}

labels = {
    'TAU' : TAU,
    'MUON': MUON,
    'ELEC': ELEC,
    'JET' : JET
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
    metafile = os.path.join(base,'meta.yml')
    if not os.path.isfile(metafile):
        print "Metadata %s not found!"%metafile
        return None
    try:
        metafile = open(metafile)
        meta = yaml.load(metafile)
        metafile.close()
        datatype = meta['type'].upper()
        classname = meta['class'].upper()
        if type(meta['weight']) is str:
            weight = float(eval(meta['weight']))
        else:
            weight = float(meta['weight'])
        treename = meta['tree']
        labelname = meta['label']
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
    if not labels.has_key(labelname):
        print "Label %s is not defined!"%labelname
        if len(labels) > 0:
            print "Use one of these:"
            for key in labels.keys():
                print key
        else:
            print "No labels have been defined!"
    labeltype = labels[labelname]
    dirs = glob.glob(os.path.join(base,'*'))
    actualdirs = []
    for dir in dirs:
        if os.path.isdir(dir):
            actualdirs.append(dir)
    files = []
    samplename = name
    if datatype == types['DATA']:
        # check for duplicate runs and take last edition
        runs = {}
        versions = {}
        for dir in actualdirs:
            datasetname = os.path.basename(dir)
            match = re.match(datapattern,datasetname)
            if not match:
                print "Warning: directory %s is not a valid dataset name!"%datasetname
            else:
                versions[match.group('version')] = None
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
        if periods:
            samplename += "_%s"%("".join(periods))
        if len(versions) > 1:
            print "Warning: multiple versions of TauD3PDMaker used:"
            for key in versions.keys():
                print key
    else:
        for dir in actualdirs:
            files += glob.glob(os.path.join(dir,'*root*'))
    return Dataset(samplename,labeltype,datatype,classtype,treename,weight,files)
