import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
from operator import add, itemgetter
import uuid
from ..tree.filtering import *
from . import multilogging
import logging
import traceback
import signal
from rootpy.io import open as ropen

class Student(Process):

    def __init__(self, name, fileset, output_queue, logging_queue, gridmode=False, **kwargs):
        
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
        self.gridmode = gridmode
        self.kwargs = kwargs
        self.output = None
                
    def run(self):
        
        # ignore sigterm signal and let parent process take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        h = multilogging.QueueHandler(self.logging_queue)
        self.logger = logging.getLogger('Student')
        self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG)

        if not self.gridmode:
            sys.stdout = multilogging.stdout(self.logger)
            sys.stderr = multilogging.stderr(self.logger)

        try:
            filename = 'student-%s-%s.root' % (self.name, self.uuid)
            with ropen(os.path.join(os.getcwd(), filename), 'recreate') as self.output:
                ROOT.gROOT.SetBatch(True)
                self.logger.info("Received %i files for processing" % len(self.fileset.files))
                self.output.cd()
                # work() is responsible for calling Write() on all objects
                self.work()
                self.output_queue.put((self.uuid, [self.event_filters, self.object_filters, self.output.GetName()]))
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.output_queue.put((self.uuid, None))
        
        self.output_queue.close()
        self.logging_queue.close()
    
    def work(self):
        """
        You must implement this method in your Student-derived class
        """ 
        raise NotImplementedError
