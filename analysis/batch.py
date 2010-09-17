import ROOT
import time
import os
import sys
from multiprocessing import Process, Pipe
import uuid
from filtering import *
ROOT.gROOT.SetBatch()

class Supervisor(object):

    def __init__(self,files,nstudents,process,name="output",nevents=-1,verbose=False):
        
        self.files = files
        self.nstudents = nstudents
        self.process = process
        self.name = name
        self.nevents = nevents
        self.verbose = verbose
        self.students = []
        self.goodStudents = []
        self.procs = []
        if os.path.exists("supervisor.log"):
            os.unlink("supervisor.log")
        self.log = open("supervisor.log","w")

    def __del__(self):

        self.log.close()
    
    def apply_for_grant(self):

        self.log.write("Will run on %i files:\n"%len(self.files))
        for file in self.files:
            self.log.write("%s\n"%file)
        # make and fill TChain
        chains = [[] for i in range(self.nstudents)]

        while len(self.files) > 0:
            for chain in chains:
                if len(self.files) > 0:
                    chain.append(self.files.pop(0))
                else:
                    break

        self.students = dict([(self.process(chain,numEvents=self.nevents),Pipe()) for chain in chains])
        self.procs = dict([(Process(target=self.__run__,args=(student,cpipe)),student) for student,(ppipe,cpipe) in self.students.items()])
    
    def supervise(self):
        
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
            self.finalize(merge=False)
            sys.exit(1)

    def publish(self,merge=True):
        
        if len(self.goodStudents) > 0:
            outputs = ["%s.root"%student.name for student in self.goodStudents]
            filters = [ppipe.recv() for ppipe in [self.students[student][0] for student in self.goodStudents]]
            self.log.write("===== Cut-flow of event filters: ====\n")
            for i in range(len(filters[0])):
                self.log.write("%s\n"%reduce(lambda x,y: x+y,[filter[i] for filter in filters]))
            if merge:
                os.system("hadd -f %s.root %s"%(self.name," ".join(outputs)))
            for output in outputs:
                os.unlink(output)

    def __run__(self,student,pipe):
    
        so = se = open("%s.log"%student.name, 'w', 0)
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        os.nice(10)

        student.coursework()
        while student.research(): pass
        student.defend()
        pipe.send(student.filters)

class Student(object):

    def __init__(self):

        self.name = uuid.uuid4().hex
        self.output = ROOT.TFile.Open("%s.root"%self.name,"recreate")
        self.filters = FilterList()
        
    def coursework(self): pass

    def research(self): pass

    def defend(self):

        self.output.Write()
        self.output.Close()
