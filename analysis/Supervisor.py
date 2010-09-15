import ROOT
import time
import copy
import os
import sys
from multiprocessing import Process

class Supervisor(object):

    def __init__(self,files,nstudents,process,name="output",verbose=False):
        
        self.files = files
        self.nstudents = nstudents
        self.process = process
        self.name = name
        self.verbose = verbose
        self.students = []
        self.procs = []
    
    def initialize(self):

        # make and fill TChain
        chains = [ROOT.TChain( "tauPerf" ) for i in range(self.nstudents)]

        while len(self.files) > 0:
            for chain in chains:
                if len(self.files) > 0:
                    chain.AddFile(self.files.pop(0))
                else:
                    break

        self.students = [self.process(chain) for chain in chains]
        self.procs = [Process(target=self.__run__,args=(student,)) for student in self.students]
    
    def execute(self):
        
        lprocs = copy.copy(self.procs)
        try:
            for p in self.procs:
                p.start()
            while len(lprocs) > 0:
                for p in lprocs:
                    if not p.is_alive():
                        p.join()
                        lprocs.remove(p)
                time.sleep(1)
        except KeyboardInterrupt:
            print "Cleaning up..."
            for p in lprocs:
                p.terminate()
            self.finalize(merge=False)
            sys.exit(1)

    def finalize(self,merge=True):
        
        outputs = ["%s.root"%student.name for student in self.students]
        logs = ["%s.log"%student.name for student in self.students]
        if merge:
            os.system("hadd %s.root %s"%(self.name," ".join(outputs)))
        for output in outputs:
            os.unlink(output)

    def __run__(self,proc):
    
        so = se = open("%s.log"%proc.name, 'w', 0)
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        os.nice(10)

        proc.initialize()
        for i in xrange(proc.tree.GetEntries()):
            proc.execute(i)
        proc.finalize()
