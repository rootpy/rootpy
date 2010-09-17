#!/usr/bin/env python

import sys
import os
import datasets
import ROOT
from TauProcessor import *
from Supervisor import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--type", action="store", type="string", dest="dataType", default="mc",
                  help="mc or data")
parser.add_option("--run", action="store", type="string", dest="run",
                  help="run number like 105009 or 153565")
parser.add_option("-v","--verbose", action="store_true", dest="verbose",
                  help="verbose", default=False)
parser.add_option("--nproc", action="store", type="int", dest="nproc",
                  help="number of students", default=2)
parser.add_option("--nevents", action="store", type="int", dest="nevents",
                  help="number of events to process by each student", default=-1)
(options, args) = parser.parse_args()

ROOT.gROOT.ProcessLine('.L dicts.C+')

data = datasets.Data7TeV
mc = datasets.MC7TeV

if options.dataType.lower()=="mc":
    myData = mc[options.run]
elif options.dataType.lower()=="data":
    myData = data[options.run]
else:
    print "CANNOT FIND DATASET TYPE",options.dataType.lower()
    sys.exit()

filelist = myData.files

master = Supervisor(files=filelist,nstudents=options.nproc,process=TauProcessor,nevents=options.nevents,verbose=options.verbose)
master.apply_for_grant()
master.supervise()
master.publish()
