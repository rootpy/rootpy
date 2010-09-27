#!/usr/bin/env python

import sys
import os
import datasets
import ROOT
from TauProcessor import *
from ROOTPy.analysis.batch import Supervisor
from ROOTPy.ntuple import NtupleChain
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-v","--verbose", action="store_true", dest="verbose",
                  help="verbose", default=False)
parser.add_option("--nproc", action="store", type="int", dest="nproc",
                  help="number of students", default=1)
parser.add_option("--nevents", action="store", type="int", dest="nevents",
                  help="number of events to process by each student", default=-1)
(options, args) = parser.parse_args()

ROOT.gROOT.ProcessLine('.L dicts.C+')

data = []
for sample in args:
    data.append(datasets.dataset[sample])

filelist = myData.files

if options.nproc == 1:
    for dataset in data:
        student = TauProcessor(dataset.files, numEvents = options.nevents)
        student.coursework()
        while student.research(): pass
        student.defend()
else:
    master = Supervisor(datasets=data,nstudents=options.nproc,process=TauProcessor,nevents=options.nevents,verbose=options.verbose)
    master.apply_for_grant()
    master.supervise()
    master.publish()
