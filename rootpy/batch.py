import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process, Pipe
from operator import add
import uuid
from rootpy.filtering import *
from atlastools import datasets
from rootpy import routines
from rootpy import multilogging

ROOT.gROOT.SetBatch()

class Supervisor(object):

    def __init__(self, name, datasets, nstudents, process, nevents=-1, verbose=False, debug = False, **kwargs):
        
        self.debug = debug
        if self.debug:
            print self.__class__.__name__+"::__init__"
        self.name = name
        self.datasets = datasets
        self.currDataset = None
        self.nstudents = nstudents
        self.process = process
        self.nevents = nevents
        self.verbose = verbose
        self.pipes = []
        self.students = []
        self.goodStudents = []
        self.procs = []
        self.kwargs = kwargs
        self.log = None
        self.hasGrant = False

        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"%(self.name,dataset.name), self.logging_queue)
        self.listener.start()

    def apply_for_grant(self):
        
        if self.debug:
            print self.__class__.__name__+"::apply_for_grant"
        if self.log:
            self.log.close()
            self.log = None
        if len(self.datasets) == 0:
            self.pipes = []
            self.students = []
            self.procs = []
            self.hasGrant = False
            return False
        dataset = self.datasets.pop(0)
        self.log.write("Will run on %i files:\n"%len(dataset.files))
        for file in dataset.files:
            self.log.write("%s\n"%file)
        # make and fill TChain
        chains = [[] for i in range(self.nstudents)]

        while len(dataset.files) > 0:
            for chain in chains:
                if len(dataset.files) > 0:
                    chain.append(dataset.files.pop(0))
                else:
                    break

        self.pipes = [Pipe() for chain in chains]
        self.students = dict([(self.process(self.name,dataset.name,dataset.label,chain,dataset.treename,dataset.datatype,dataset.classtype,dataset.weight,numEvents=self.nevents,pipe=cpipe, debug = self.debug, **self.kwargs),ppipe) for chain,(ppipe,cpipe) in zip(chains,self.pipes)])
        self.goodStudents = []
        self.hasGrant = True
        self.currDataset = dataset
        return True
   
    def __cleanup(self):
        
        outputs = [student.outputfilename for student in self.students]
        for output in outputs:
            os.unlink(output)
        self.__logging_shutdown()

    def __logging_shutdown(self):
        
        self.logging_queue.put_nowait(None)
        self.listener.join()

    def supervise(self):
        
        if self.debug:
            print self.__class__.__name__+"::supervise"
        if self.hasGrant:
            lprocs = [p for p in self.students.keys()]
            try:
                for p in self.students.keys():
                    p.start()
                while len(lprocs) > 0:
                    for p in lprocs:
                        if not p.is_alive():
                            p.join()
                            if p.exitcode == 0:
                                self.goodStudents.append(self.students[p])
                            lprocs.remove(p)
                    time.sleep(1)
            except KeyboardInterrupt:
                print "Cleaning up..."
                for p in lprocs:
                    p.terminate()
                self.__cleanup()
                sys.exit(1)

    def publish(self,merge=True):
        
        if self.debug:
            print self.__class__.__name__+"::publish"
        if len(self.goodStudents) > 0:
            outputs = [student.outputfilename for student in self.goodStudents]
            filters = [pipe.recv() for pipe in [self.students[student] for student in self.goodStudents]]
            self.log.write("===== Cut-flow of event filters for dataset %s: ====\n"%(self.currDataset.name))
            totalEvents = 0
            for i in range(len(filters[0])):
                totalFilter = reduce(add,[filter[i] for filter in filters])
                if i == 0:
                    totalEvents = totalFilter.total
                self.log.write("%s\n"%totalFilter)
            if merge:
                os.system("hadd -f %s.root %s"%(self.currDataset.name," ".join(outputs)))
            for output in outputs:
                os.unlink(output)
            # set weights:
            if totalEvents != 0 and self.currDataset.datatype != datasets.types['DATA']:
                file = ROOT.TFile.Open("%s.root"%self.currDataset.name,"update")
                trees = routines.getTrees(file)
                for tree in trees:
                    tree.SetWeight(self.currDataset.weight/totalEvents)
                    tree.Write("",ROOT.TObject.kOverwrite)
                file.Close()
        self.__logging_shutdown()

class Student(Process):

    def __init__(self, processname, name, label, files, treename, datatype, classtype, weight, numEvents, pipe, debug = False):
        
        Process.__init__(self)
        self.debug = debug
        if self.debug:
            print self.__class__.__name__+"::__init__"
        self.uuid = uuid.uuid4().hex
        self.filters = FilterList()
        self.processname = name
        self.name = name
        self.label = label
        self.files = files
        self.treename = treename
        self.datatype = datatype
        self.classtype = classtype
        self.weight = weight
        self.numEvents = numEvents
        self.event = 0
        self.pipe = pipe
        self.outputfilename = "student-%s-%s.root"%(self.processname,self.uuid)
        self.output = ROOT.TFile.Open(self.outputfilename,"recreate")

    def run(self):
       
        #so = se = open(student.logfilename, 'w', 0)
        #os.dup2(so.fileno(), sys.stdout.fileno())
        #os.dup2(se.fileno(), sys.stderr.fileno())
        sys.stdout = StdOut()
        sys.stderr = StdErr()
        if self.debug:
            print self.__class__.__name__+"::__run__"
        os.nice(10)
        self.coursework()
        self.research()
        self.defend()
        
    def coursework(self):
        
        if self.debug:
            print self.__class__.__name__+"::coursework"
        
    def research(self):

        if self.debug:
            print self.__class__.__name__+"::research"

    def defend(self):
        
        if self.debug:
            print self.__class__.__name__+"::defend"
        self.pipe.send(self.filters)
        self.output.Write()
        self.output.Close()
