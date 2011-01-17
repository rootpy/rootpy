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
import logging
import traceback

ROOT.gROOT.SetBatch()

class Supervisor(Process):

    def __init__(self, name, fileset, nstudents, process, nevents = -1, **kwargs):
        
        Process.__init__(self) 
        self.name = name
        self.fileset = fileset
        self.nstudents = nstudents
        self.process = process
        self.nevents = nevents
        self.pipes = []
        self.students = []
        self.good_students = []
        self.kwargs = kwargs
        
    def run(self):
        
        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"% (self.name, self.fileset.name), self.logging_queue)
        self.listener.start()
        
        h = multilogging.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        
        logger = logging.getLogger(self.name)
        sys.stdout = multilogging.stdout(logger)
        sys.stderr = multilogging.stderr(logger)
       
        try:
            self.__apply_for_grant()
            self.__supervise()
            self.__publish()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.__logging_shutdown()
        self.__logging_shutdown()
    
    def __apply_for_grant(self):
        
        print "Will run on %i files:"% len(self.fileset.files)
        for filename in self.fileset.files:
            print "%s"% filename
        
        filesets = self.fileset.split(self.nstudents)
        
        self.pipes = [Pipe() for i in xrange(self.nstudents)]
        self.students = dict([(
            self.process(
                name = self.name,
                fileset = self.fileset,
                nevents = self.nevents,
                pipe = cpipe,
                logging_queue = self.logging_queue,
                **self.kwargs
            ), ppipe) for fileset,(ppipe,cpipe) in zip(filesets,self.pipes)])
        self.good_students = []
   
    def __cleanup(self):
        
        outputs = [student.outputfilename for student in self.students]
        for output in outputs:
            os.unlink(output)

    def __logging_shutdown(self):
        
        self.logging_queue.put_nowait(None)
        self.listener.join()

    def __supervise(self):
        
        lprocs = [p for p in self.students.keys()]
        for p in self.students.keys():
            p.start()
        while len(lprocs) > 0:
            for p in lprocs:
                if not p.is_alive():
                    p.join()
                    if p.exitcode == 0:
                        self.good_students.append(self.students[p])
                    lprocs.remove(p)
            time.sleep(1)

    def __publish(self, merge = True):
        
        if len(self.good_students) > 0:
            outputs = [student.outputfilename for student in self.good_students]
            filters = [pipe.recv() for pipe in [self.students[student] for student in self.good_students]]
            print "===== Cut-flow of event filters for dataset %s: ====\n"% self.fileset.name
            totalEvents = 0
            for i in range(len(filters[0])):
                totalFilter = reduce(add,[filter[i] for filter in filters])
                if i == 0:
                    totalEvents = totalFilter.total
                self.log.write("%s\n"%totalFilter)
            if merge:
                os.system("hadd -f %s.root %s"%(self.fileset.name, " ".join(outputs)))
            for output in outputs:
                os.unlink(output)
            # set weights:
            if totalEvents != 0 and self.fileset.datatype != datasets.types['DATA']:
                outfile = ROOT.TFile.Open("%s.root"% self.fileset.name, "update")
                trees = routines.getTrees(outfile)
                for tree in trees:
                    tree.SetWeight(self.fileset.weight/totalEvents)
                    tree.Write("", ROOT.TObject.kOverwrite)
                outfile.Close()

class Student(Process):

    def __init__(self, name, fileset, nevents, pipe, logging_queue, **kwargs):
        
        Process.__init__(self)
        self.uuid = uuid.uuid4().hex
        self.filters = FilterList()
        self.name = name
        self.nevents = nevents
        self.event = 0
        self.pipe = pipe
        self.outputfilename = "student-%s-%s.root"% (self.name, self.uuid)
        self.output = ROOT.TFile.Open(self.outputfilename, "recreate")

        # user-defined attrs
        for key, value in kwargs.items():
            # need to make this safer
            setattr(self, key, value)

        # logging
        h = multilogging.QueueHandler(logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    def run(self):
        
        logger = logging.getLogger(self.uuid)
        sys.stdout = multilogging.stdout(logger)
        sys.stderr = multilogging.stderr(logger)
        self.coursework()
        self.research()
        self.defend()
        
    def coursework(self): pass
        
    def research(self): pass

    def defend(self):
        
        self.pipe.send(self.filters)
        self.output.Write()
        self.output.Close()
