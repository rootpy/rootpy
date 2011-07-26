import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
from operator import add, itemgetter
import uuid
from ..tree.filtering import *
from atlastools import datasets
from .. import routines
from .. import multilogging
import logging
import traceback

class Supervisor(Process):

    def __init__(self, name, outputname, fileset, nstudents, process, args = None, **kwargs):
        
        Process.__init__(self) 
        self.name = name
        self.fileset = fileset
        self.outputname = outputname
        self.nstudents = min(nstudents, len(fileset.files))
        self.process = process
        self.students = []
        self.good_students = []
        self.student_outputs = []
        self.kwargs = kwargs
        self.logger = None
        self.args = args
        
    def run(self):

        ROOT.gROOT.SetBatch()
        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"% (self.name, self.outputname), self.logging_queue)
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
        print "Done"
        self.logging_queue.put(None)
        self.listener.join()

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
        self.process_table = dict([(p.uuid, p) for p in self.students])
        self.good_students = []
            
    def __supervise(self):
        
        students = self.students[:]
        for p in students:
            p.start()
        nstudents = len(students)
        while students:
            while not self.output_queue.empty():
                id, output = self.output_queue.get()
                process = self.process_table[id]
                process.join()
                students.remove(process)
                if output is not None and process.exitcode == 0:
                    self.good_students.append(process)
                    self.student_outputs.append(output)
            time.sleep(1)

    def __publish(self, merge = True):
        
        if len(self.good_students) > 0:
            outputs = []
            event_filters = []
            object_filters = []
            for thing in self.student_outputs:
                event_filters.append(thing[0])
                object_filters.append(thing[1])
                outputs.append(thing[2])
            print "===== Cut-flow of event filters for dataset %s: ====\n"% self.outputname
            totalEvents = 0
            combinedFilterlist = reduce(FilterList.merge, event_filters)
            if len(combinedFilterlist) > 0:
                totalEvents = combinedFilterlist[0].total
            print ": Event Filters\n%s"% combinedFilterlist
            combinedFilterlist = reduce(FilterList.merge, object_filters)
            print ": Object Filters\n%s"% combinedFilterlist
            """
                for filter in len(object_filters[0]):
                    filters = map(itemgetter(filter), object_filters)
                    combinedFilterlist = reduce(FilterList.merge, filters)
                    print combinedFilterlist
            """
            if merge:
                os.system("hadd -f %s.root %s"%(self.outputname, " ".join(outputs)))
            for output in outputs:
                os.unlink(output)
            # set weights:
            if totalEvents != 0 and self.fileset.datatype != datasets.types['DATA']:
                outfile = ROOT.TFile.Open("%s.root"% self.outputname, "update")
                trees = routines.getTrees(outfile)
                for tree in trees:
                    tree.SetWeight(self.fileset.weight/totalEvents)
                    tree.Write("", ROOT.TObject.kOverwrite)
                outfile.Close()


