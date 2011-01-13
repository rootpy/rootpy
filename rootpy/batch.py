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
        self.logger = None
        
    def run(self):
        
        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"% (self.name, self.fileset.name), self.logging_queue)
        self.listener.start()
        
        h = multilogging.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        
        self.logger = logging.getLogger("Supervisor")
        sys.stdout = multilogging.staged_stdout(self.logger)
        sys.stderr = multilogging.staged_stderr(self.logger)
       
        try:
            self.__apply_for_grant()
            self.__supervise()
            self.__publish()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.__terminate()
        self.__logging_shutdown()
    
    def __apply_for_grant(self):
        
        print "Will run on %i files:"% len(self.fileset.files)
        for filename in self.fileset.files:
            print "%s"% filename
        sys.stdout.flush()
        
        filesets = self.fileset.split(self.nstudents)
        
        self.pipes = [Pipe() for i in xrange(self.nstudents)]
        self.students = dict([(
            self.process(
                name = self.name,
                fileset = fileset,
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

    def __terminate(self):

        for s in self.students:
            s.join()
    
    def __supervise(self):
        
        lprocs = self.students.keys()[:]
        for p in self.students.keys():
            p.start()
        while len(lprocs) > 0:
            for p in lprocs:
                if not p.is_alive():
                    p.join()
                    if p.exitcode == 0:
                        self.good_students.append(p)
                    lprocs.remove(p)
            time.sleep(1)

    def __publish(self, merge = True):
        
        if len(self.good_students) > 0:
            outputs = [student.outputfilename for student in self.good_students]
            filters = []
            for pipe in [self.students[student] for student in self.good_students]:
                filters.append(pipe.recv())
            print "===== Cut-flow of event filters for dataset %s: ====\n"% self.fileset.name
            totalEvents = 0
            for i in range(len(filters[0])):
                totalFilter = reduce(add,[filter[i] for filter in filters])
                if i == 0:
                    totalEvents = totalFilter.total
                print totalFilter
            sys.stdout.flush()
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
        self.fileset = fileset
        self.nevents = nevents
        self.event = 0
        self.pipe = pipe
        self.logging_queue = logging_queue
        self.outputfilename = "student-%s-%s.root"% (self.name, self.uuid)
        self.output = ROOT.TFile.Open(self.outputfilename, "recreate")
        self.logger = None
        
    def run(self):
        
        # logging
        h = multilogging.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("Student")
        sys.stdout = multilogging.stdout(self.logger)
        sys.stderr = multilogging.stderr(self.logger)
        self.logger.info("Received %i files for processing"% len(self.fileset.files))
        try:
            self.coursework()
            self.research()
            self.defend()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.pipe.close()
            raise
        
    def coursework(self): pass
        
    def research(self): pass

    def defend(self):
        
        self.pipe.send(self.filters)
        self.pipe.close()
        self.output.Write()
        self.output.Close()
