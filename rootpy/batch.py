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

class Supervisor(Process):

    def __init__(self, name, fileset, nstudents, process, nevents = -1, **kwargs):
        
        Process.__init__(self) 
        self.debug = debug
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
        self.listener = multilogging.Listener("supervisor-%s-%s.log"%(self.name,dataset.name), self.logging_queue)
        self.listener.start()
        
        self.apply_for_grant()
        self.supervise()
        self.publish()
    
    def __apply_for_grant(self):
        
        self.log.write("Will run on %i files:\n"%len(self.fileset.files))
        for filename in self.fileset.files:
            self.log.write("%s\n"%filename)
        
        filesets = self.fileset.split(self.nstudents)
        
        self.pipes = [Pipe() for chain in chains]
        self.students = dict([(
            self.process(
                fileset,
                self.name,
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
        self.__logging_shutdown()

    def __logging_shutdown(self):
        
        self.logging_queue.put_nowait(None)
        self.listener.join()

    def __supervise(self):
        
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
                                self.good_students.append(self.students[p])
                            lprocs.remove(p)
                    time.sleep(1)
            except KeyboardInterrupt:
                print "Cleaning up..."
                for p in lprocs:
                    p.terminate()
                self.__cleanup()
                sys.exit(1)

    def __publish(self, merge = True):
        
        if len(self.good_students) > 0:
            outputs = [student.outputfilename for student in self.good_students]
            filters = [pipe.recv() for pipe in [self.students[student] for student in self.good_students]]
            self.log.write("===== Cut-flow of event filters for dataset %s: ====\n"% self.fileset.name)
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
        self.__logging_shutdown()

class Student(Process):

    def __init__(self, name, dataset, nevents, pipe, logging_queue):
        
        Process.__init__(self)
        self.uuid = uuid.uuid4().hex
        self.filters = FilterList()
        self.name = name
        self.nevents = nevents
        self.event = 0
        self.pipe = pipe
        self.outputfilename = "student-%s-%s.root"% (self.processname,self.uuid)
        self.output = ROOT.TFile.Open(self.outputfilename, "recreate")

    def run(self):
       
        sys.stdout = StdOut()
        sys.stderr = StdErr()
        self.coursework()
        self.research()
        self.defend()
        
    def coursework(self): pass
        
    def research(self): pass

    def defend(self):
        
        self.pipe.send(self.filters)
        self.output.Write()
        self.output.Close()
