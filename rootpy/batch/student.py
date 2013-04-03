# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
import os
import sys
import multiprocessing
from multiprocessing import Process
import uuid

from ..logger import multilogging
from .. import log; log = log[__name__]
import logging
try:
    logging.captureWarnings(True)
except AttributeError:
    pass

import traceback
import signal
from rootpy.io import root_open
import cProfile as profile
import subprocess


class Student(Process):

    def __init__(self,
            name,
            files,
            output_queue,
            logging_queue,
            gridmode=False,
            metadata=None,
            profile=False,
            nice=0,
            **kwargs):

        Process.__init__(self)
        self.uuid = uuid.uuid4().hex
        self.filters = {}
        self.name = name
        self.files = files
        self.metadata = metadata
        self.logging_queue = logging_queue
        self.output_queue = output_queue
        self.output = None
        self.gridmode = gridmode
        self.nice = nice
        self.kwargs = kwargs
        self.output = None
        self.queuemode = isinstance(files, multiprocessing.queues.Queue)
        self.profile = profile

    def run(self):

        # ignore sigterm signal and let parent process take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        ROOT.gROOT.SetBatch()

        os.nice(self.nice)

        h = multilogging.QueueHandler(self.logging_queue)
        # get the top-level logger
        log_root = logging.getLogger()
        # clear any existing handlers in the top-level logger
        log_root.handlers = []
        # add the queuehandler
        log_root.addHandler(h)

        if not self.gridmode:
            # direct stdout and stderr to the local logger
            sys.stdout = multilogging.stdout(log)
            sys.stderr = multilogging.stderr(log)

        try:
            filename = 'student-%s-%s.root' % (self.name, self.uuid)
            with root_open(os.path.join(
                    os.getcwd(), filename), 'recreate') as self.output:
                ROOT.gROOT.SetBatch(True)
                if self.queuemode:
                    log.info("Receiving files from Supervisor's queue")
                else:
                    log.info(
                            "Received %i files from Supervisor for processing" %
                            len(self.files))
                self.output.cd()
                if self.profile:
                    profile_filename = 'student-%s-%s.profile' % (
                            self.name, self.uuid)
                    profile.runctx('self.work()',
                            globals=globals(),
                            locals=locals(),
                            filename=profile_filename)
                    self.output_queue.put(
                            (self.uuid,
                                [self.filters,
                                 self.output.GetName(),
                                 profile_filename]))
                else:
                    self.work()
                    self.output_queue.put(
                            (self.uuid,
                                [self.filters, self.output.GetName()]))
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            self.output_queue.put((self.uuid, None))

        self.output_queue.close()
        self.logging_queue.close()

    @staticmethod
    def merge(inputs, output, metadata):
        """
        Default merging mechanism.
        Override this method to define merging behaviour suitable
        to your needs.
        """
        subprocess.call(['hadd', output + '.root'] + inputs)

    def work(self):
        """
        You must implement this method in your Student-derived class
        """
        raise NotImplementedError(
                "implement this method in your Student-derived class")
