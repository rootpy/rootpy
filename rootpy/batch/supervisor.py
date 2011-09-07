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
import shutil
import subprocess
import signal
try:
    import cPickle as pickle
except:
    import pickle


class Supervisor(Process):

    def __init__(self, name, outputname, fileset, nstudents, process, connect_queue, gridmode=False, args=None, **kwargs):
                
        Process.__init__(self) 
        self.name = name
        self.fileset = fileset
        self.outputname = outputname
        self.gridmode = gridmode
        if self.gridmode:
            self.nstudents = 1
        else:
            self.nstudents = min(nstudents, len(fileset.files))
        self.process = process
        self.student_outputs = []
        self.kwargs = kwargs
        self.logger = None
        self.args = args
        self.connect_queue = connect_queue
        
    def run(self):
        
        # ignore sigterm signal and let parent take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        ROOT.gROOT.SetBatch()
        
        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log"% (self.name, self.outputname), self.logging_queue)
        self.listener.start()
        
        h = multilogging.QueueHandler(self.logging_queue)
        self.logger = logging.getLogger("Supervisor")
        self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.gridmode:
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
        self.output_queue.close()
        self.logging_queue.put(None)
        self.listener.join()

    def __apply_for_grant(self):
        
        print "Will run on %i files:"% len(self.fileset.files)
        for filename in self.fileset.files:
            print "%s"% filename
        sys.stdout.flush()
        filesets = self.fileset.split(self.nstudents)
        self.output_queue = multiprocessing.Queue(-1)
        students = [
            self.process(
                name = self.name,
                fileset = fileset,
                output_queue = self.output_queue,
                logging_queue = self.logging_queue,
                gridmode = self.gridmode,
                args = self.args,
                **self.kwargs
            ) for fileset in filesets ]
        self.process_table = dict([(p.uuid, p) for p in students])
            
    def __supervise(self):
        
        for student in self.process_table.values():
            student.start()
        while self.process_table:
            if not self.connect_queue.empty():
                msg = self.connect_queue.get()
                if msg is None:
                    print "will now terminate..."
                    for student in self.process_table.values():
                        student.terminate()
                    return
            while not self.output_queue.empty():
                id, output = self.output_queue.get()
                process = self.process_table[id]
                process.join()
                del self.process_table[id]
                if output is not None and process.exitcode == 0:
                    self.student_outputs.append(output)
            time.sleep(1)
    
    def __publish(self, merge = True):
        
        if len(self.student_outputs) > 0:
            outputs = []
            event_filters = []
            object_filters = []
            for thing in self.student_outputs:
                event_filters.append(thing[0])
                object_filters.append(thing[1])
                outputs.append(thing[2])
            print "\n===== Cut-flow of event filters for dataset %s: ====\n"% self.outputname
            totalEvents = 0
            combinedEventFilterlist = reduce(FilterList.merge, event_filters)
            if len(combinedEventFilterlist) > 0:
                totalEvents = combinedEventFilterlist[0].total
            print "Event Filters:\n%s"% combinedEventFilterlist
            combinedObjectFilterlist = reduce(FilterList.merge, object_filters)
            print "Object Filters:\n%s"% combinedObjectFilterlist
            pfile = open("cutflow.p",'w')
            pickle.dump({"event": combinedEventFilterlist,
                         "object": combinedObjectFilterlist}, pfile)
            pfile.close()
            if merge:
                outputname = "%s.root" % self.outputname 
                if os.path.exists(outputname):
                    os.unlink(outputname)
                if len(outputs) == 1:
                    shutil.move(outputs[0], outputname)
                else:
                    subprocess.call(["hadd", outputname] + outputs)
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
