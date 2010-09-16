import ROOT
import time
import os
import sys
from multiprocessing import Process

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
    
    def apply_for_grant(self):

        # make and fill TChain
        chains = [[] for i in range(self.nstudents)]

        while len(self.files) > 0:
            for chain in chains:
                if len(self.files) > 0:
                    chain.append(self.files.pop(0))
                else:
                    break

        self.students = [self.process(chain,numEvents=self.nevents) for chain in chains]
        self.procs = dict([(Process(target=self.__run__,args=(student,)),student) for student in self.students])
    
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
            filters = [student.filters for student in self.goodStudents]
            for i in range(len(filters[0])):
                print reduce(lambda x,y: x+y,[filter[i] for filter in filters])
            if merge:
                os.system("hadd -f %s.root %s"%(self.name," ".join(outputs)))
            for output in outputs:
                os.unlink(output)

    def __run__(self,proc):
    
        so = se = open("%s.log"%proc.name, 'w', 0)
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        os.nice(10)

        proc.coursework()
        while proc.research(): pass
        proc.defend()
