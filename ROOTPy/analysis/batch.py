import ROOT
import time
import os
import sys
from multiprocessing import Process, Pipe
import uuid
from filtering import *
ROOT.gROOT.SetBatch()

class Supervisor(object):

    def __init__(self,datasets,nstudents,process,nevents=-1,verbose=False,**kwargs):
        
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

    def __del__(self):

        self.log.close()
    
    def apply_for_grant(self):

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
        self.log = open("supervisor-"+dataset.name+".log","w",0)
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
        self.students = dict([(self.process(chain,dataset.treename,dataset.datatype,dataset.classname,dataset.weight,numEvents=self.nevents,pipe=cpipe,**self.kwargs),ppipe) for chain,(ppipe,cpipe) in zip(chains,self.pipes)])
        self.procs = dict([(Process(target=self.__run__,args=(student,)),student) for student in self.students])
        self.goodStudents = []
        self.hasGrant = True
        self.currDataset = dataset
        return True
    
    def supervise(self):
        
        if self.hasGrant:
            lprocs = [p for p in self.procs.keys()]
            try:
                for p in self.procs.keys():
                    p.start()
                while len(lprocs) > 0:
                    for p in lprocs:
                        if not p.is_alive():
                            p.join()
                            if p.exitcode == 0:
                                self.goodStudents.append(self.procs[p])
                            lprocs.remove(p)
                    time.sleep(1)
            except KeyboardInterrupt:
                print "Cleaning up..."
                for p in lprocs:
                    p.terminate()
                self.publish(merge=False)
                sys.exit(1)

    def publish(self,merge=True):
        
        if len(self.goodStudents) > 0:
            outputs = ["%s.root"%student.name for student in self.goodStudents]
            filters = [pipe.recv() for pipe in [self.students[student] for student in self.goodStudents]]
            self.log.write("===== Cut-flow of event filters for dataset %s: ====\n"%(self.currDataset.tag))
            for i in range(len(filters[0])):
                self.log.write("%s\n"%reduce(lambda x,y: x+y,[filter[i] for filter in filters]))
            if merge:
                os.system("hadd -f %s.root %s"%(self.currDataset.tag," ".join(outputs)))
            for output in outputs:
                os.unlink(output)
        if self.log:
            self.log.close()
            self.log = None

    def __run__(self,student):
    
        so = se = open("%s.log"%student.name, 'w', 0)
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        #os.nice(10)
        student.coursework()
        while student.research(): pass
        student.defend()

class Student(object):

    def __init__(self,files,treename,weight,numEvents,pipe):

        self.name = uuid.uuid4().hex
        self.output = ROOT.TFile.Open("%s.root"%self.name,"recreate")
        self.filters = FilterList()
        self.files = files
        self.treename = treename
        self.weight = weight
        self.numEvents = numEvents
        self.event = 0
        self.pipe = pipe
        
    def coursework(self): pass

    def research(self): pass

    def defend(self):
        
        self.pipe.send(self.filters)
        self.output.Write()
        self.output.Close()
