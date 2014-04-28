# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

import os
import sys
import multiprocessing
from multiprocessing import Process
import uuid
import traceback
import signal
import cProfile as profile
import subprocess
import logging
try:
    logging.captureWarnings(True)
except AttributeError:
    pass

from .. import log; log = log[__name__]
from ..logger import multilogging
from ..io import root_open

__all__ = [
    'Student',
]


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

        super(Student, self).__init__()
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
        if self.gridmode:
            self.queuemode = False
        else:
            self.queuemode = isinstance(files, multiprocessing.queues.Queue)
        self.profile = profile

    def __repr__(self):
        return '{0}(id={1})'.format(self.name, self.uuid)

    def run(self):

        if not self.gridmode:
            # ignore sigterm signal and let the supervisor process handle this
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            h = multilogging.QueueHandler(self.logging_queue)
            # get the top-level logger
            log_root = logging.getLogger()
            # clear any existing handlers in the top-level logger
            log_root.handlers = []
            # add the queuehandler
            log_root.addHandler(h)
            # direct stdout and stderr to the local logger
            sys.stdout = multilogging.stdout(log)
            sys.stderr = multilogging.stderr(log)

        ROOT.gROOT.SetBatch()

        os.nice(self.nice)

        try:
            filename = 'student-{0}-{1}.root'.format(
                self.name, self.uuid)
            with root_open(os.path.join(
                    os.getcwd(), filename), 'recreate') as self.output:
                if self.queuemode:
                    log.info("Receiving files from Supervisor's queue")
                else:
                    log.info(
                        "Received {0:d} files from Supervisor "
                        "for processing".format(
                            len(self.files)))
                self.output.cd()
                if self.profile:
                    profile_filename = 'student-{0}-{1}.profile'.format(
                        self.name, self.uuid)
                    profile.runctx(
                        'self.work()',
                        globals=globals(),
                        locals=locals(),
                        filename=profile_filename)
                    result = (
                        self.uuid,
                        [self.filters,
                         self.output.GetName(),
                         profile_filename])
                else:
                    self.work()
                    result = (
                        self.uuid,
                        [self.filters,
                         self.output.GetName()])
        except:
            if self.gridmode:
                raise
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
            result = (self.uuid, None)

        if self.gridmode:
            id, result = result
            self.output_queue.append(result)
        else:
            self.output_queue.put(result)
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
