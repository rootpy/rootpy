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

class Student(Process):

    def __init__(self, name, fileset, output_queue, logging_queue, **kwargs):
        
        Process.__init__(self)
        self.uuid = uuid.uuid4().hex
        self.event_filters = EventFilterList()
        self.object_filters = ObjectFilterList()
        self.name = name
        self.fileset = fileset
        self.logging_queue = logging_queue
        self.output_queue = output_queue
        self.logger = None
        self.output = None
                
    def run(self):
        
        h = multilogging.QueueHandler(self.logging_queue)
        self.logger = logging.getLogger("Student")
        self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG)
        sys.stdout = multilogging.stdout(self.logger)
        sys.stderr = multilogging.stderr(self.logger)

        try:
            filename = "student-%s-%s.root"% (self.name, self.uuid)
            self.output = ROOT.TFile.Open(os.path.join(os.getcwd(),filename), "recreate")
            ROOT.gROOT.SetBatch(True)
            self.logger.info("Received %i files for processing"% len(self.fileset.files))
            self.output.cd()
            self.coursework()
            self.research()
            self.output.Write()
            self.output.Close()
            self.output_queue.put((self.uuid, [self.event_filters, self.object_filters, self.output.GetName()]))
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.output_queue.put((self.uuid, None))
        self.output_queue.close()
        self.logging_queue.close()
    
    def coursework(self):
        
        raise NotImplementedError
        
    def research(self):

        raise NotImplementedError
