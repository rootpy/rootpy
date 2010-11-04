#!/usr/bin/env python

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-v","--verbose", action="store_true", dest="verbose",
                  help="verbose", default=False)
parser.add_option("--nproc", action="store", type="int", dest="nproc",
                  help="number of students", default=1)
parser.add_option("--nevents", action="store", type="int", dest="nevents",
                  help="number of events to process by each student", default=-1)
parser.add_option("--jes", action="store_true", dest="doJESsys",
                  help="recalculate affected variables at EM+JES", default=False)
parser.add_option("--grl", action="store", type="str", dest="grl",
                  help="good runs list", default=None)
parser.add_option('-p',"--periods", action="store", type="str", dest="periods",
                  help="data period", default=None)
(options, args) = parser.parse_args()

import sys
import os
import datasets
import ROOT
from TauProcessor import *
from ROOTPy.analysis.batch import Supervisor
from ROOTPy.ntuple import NtupleChain
from ROOTPy.datasets import *

ROOT.gROOT.ProcessLine('.L dicts.C+')

if options.periods:
    options.periods = options.periods.split(',')

if len(args) == 0:
    print "No samples specified!"
    sys.exit(1)

data = []
for sample in args:
    dataset = get_sample(sample,options.periods)
    if not dataset:
        print "FATAL: sample %s does not exist!"%sample
        sys.exit(1)
    data.append(dataset)

if options.nproc == 1:
    for dataset in data:
        student = TauProcessor(dataset.files, numEvents = options.nevents, doJESsys=options.doJESsys, grl=options.grl)
        student.coursework()
        while student.research(): pass
        student.defend()
else:
    supervisor = Supervisor(datasets=data,nstudents=options.nproc,process=TauProcessor,nevents=options.nevents,verbose=options.verbose,doJESsys=options.doJESsys,grl=options.grl)
    while supervisor.apply_for_grant():
        supervisor.supervise()
        supervisor.publish()
