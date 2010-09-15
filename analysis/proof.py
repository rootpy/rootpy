import ROOT, sys, os
import datasets
import sys

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--type", action="store", type="string", dest="dataType", default="mc",
                  help="mc or data")
parser.add_option("--run", action="store", type="string", dest="run",
                  help="run number like 105009 or 153565")
parser.add_option("--proof", action="store_true", dest="doProof",
                  help="use proof", default=False)
parser.add_option("-v","--verbose", action="store_true", dest="verbose",
                  help="verbose", default=False)
parser.add_option("--numProc", action="store", type="int", dest="numProc",
                  help="number of parallel proof slaves", default=2)

(options, args) = parser.parse_args()

ROOT.gSystem.CompileMacro( 'EMJESfix.hpp')
ROOT.gROOT.ProcessLine('.L dicts.C+')

print options.dataType.lower(), options.run

data = datasets.Data7TeV
mc = datasets.MC7TeV

print "Available MC datasets",mc.keys()
print "Available DATA datasets",data.keys()

if options.dataType.lower()=="mc":
    myData = mc[options.run]
elif options.dataType.lower()=="data":
    myData = data[options.run]
else:
    print "CANNOT FIND DATASET TYPE",options.dataType.lower()
    sys.exit()

###
crossWeight = 0.
filedict = {}
filelist = []
print "Add ", myData.tag,"\t-\t",myData.runnumber,"\t:\t",len(myData.files)," files "
filelist = myData.files

# make and fill TChain
trees = { 'r' : ROOT.TChain( "tauPerf" ) }
for k,v in trees.items():
    for f in filelist:
        v.AddFile( f )

packageLib = os.environ['PWD']
rootLib = os.environ['ROOTSYS'] + '/lib/root'
baseLib = os.environ['LD_LIBRARY_PATH']

if options.doProof:
    session = ROOT.TProof.Open('')
    session.SetLogLevel( 0 )
    session.SetParallel( options.numProc )

    pathSeparator = ':'
    ROOT.gProof.AddDynamicPath( packageLib )
    ROOT.gProof.AddDynamicPath( rootLib )
    ROOT.gProof.Exec( 'gSystem->Setenv("LD_LIBRARY_PATH","%s")'%( pathSeparator.join((packageLib, baseLib)) ) )
    ROOT.gProof.Exec( 'gSystem->Setenv("PYTHONPATH","%s")'%( pathSeparator.join((packageLib, rootLib)) ) )

    trees['r'].SetProof()
# Now run it!
trees['r'].Process( "TPySelector", "D3PDFlattener" )

