import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
from operator import add
import uuid
from rootpy.filtering import *
from atlastools import datasets
from rootpy import routines
from rootpy import multilogging
import logging
import traceback

class Supervisor(Process):

    def __init__(self, name, fileset, nstudents, process, args = None, **kwargs):
        
        Process.__init__(self) 
        self.name = name
        self.fileset = fileset
        self.nstudents = min(nstudents, len(fileset.files))
        self.process = process
        self.students = []
        self.good_students = []
        self.kwargs = kwargs
        self.logger = None
        self.args = args
        
    def run(self):

        ROOT.gROOT.SetBatch()
        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"% (self.name, self.fileset.name), self.logging_queue)
        self.listener.start()
        
        h = multilogging.QueueHandler(self.logging_queue)
        self.logger = logging.getLogger("Supervisor")
        self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG)
        sys.stdout = multilogging.stdout(self.logger)
        sys.stderr = multilogging.stderr(self.logger)
       
        try:
            self.__apply_for_grant()
            self.__supervise()
            self.__publish()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.terminate()
        self.__logging_shutdown()
    
    def __apply_for_grant(self):
        
        print "Will run on %i files:"% len(self.fileset.files)
        for filename in self.fileset.files:
            print "%s"% filename
        sys.stdout.flush()
        filesets = self.fileset.split(self.nstudents)
        self.output_queue = multiprocessing.Queue(-1)
        self.students = [
            self.process(
                name = self.name,
                fileset = fileset,
                output_queue = self.output_queue,
                logging_queue = self.logging_queue,
                args = self.args,
                **self.kwargs
            ) for fileset in filesets ]
        self.good_students = []

    def __logging_shutdown(self):
        
        self.logging_queue.put_nowait(None)
        self.listener.join()

    def terminate(self):

        for s in self.students:
            s.terminate()
            self.__logging_shutdown()
        Process.terminate(self)
    
    def __supervise(self):
        
        lprocs = self.students[:]
        for p in lprocs:
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
            outputs = []
            filters = []
            while not self.output_queue.empty():
                thing = self.output_queue.get()
                if isinstance(thing, basestring):
                    outputs.append(thing)
                elif isinstance(thing, FilterList):
                    filters.append(thing)
                else:
                    print "I don't know what to do with an object of type %s"% type(thing)
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

    def __init__(self, name, fileset, output_queue, logging_queue, **kwargs):
        
        Process.__init__(self)
        self.uuid = uuid.uuid4().hex
        self.filters = FilterList()
        self.name = name
        self.fileset = fileset
        self.logging_queue = logging_queue
        self.output_queue = output_queue
        self.logger = None
        self.output = None
                
    def run(self):
        
        try:
            filename = "student-%s-%s.root"% (self.name, self.uuid)
            self.output = ROOT.TFile.Open(os.path.join(os.getcwd(),filename), "recreate")
            ROOT.gROOT.SetBatch(True)
            # logging
            h = multilogging.QueueHandler(self.logging_queue)
            self.logger = logging.getLogger("Student")
            self.logger.addHandler(h)
            self.logger.setLevel(logging.DEBUG)
            sys.stdout = multilogging.stdout(self.logger)
            sys.stderr = multilogging.stderr(self.logger)
            self.logger.info("Received %i files for processing"% len(self.fileset.files))
            self.output.cd()
            self.coursework()
            self.research()
            self.defend()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.terminate()
        
    def terminate(self):
        
        try:
            self.defend()
            os.remove(self.output.GetName())
        except:
            pass
        Process.terminate(self)
    
    def coursework(self): pass
        
    def research(self): pass

    def defend(self):
        
        self.output_queue.put(self.filters)
        self.output_queue.put(self.output.GetName())
        self.output_queue.close()
        self.output.Write()
        self.output.Close()
