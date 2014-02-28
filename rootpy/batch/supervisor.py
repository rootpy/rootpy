# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
# multiprocessing uses the exceptions from the Queue module
import Queue
import traceback
import signal
import pstats
import cStringIO as StringIO
import shutil
import logging
try:
    logging.captureWarnings(True)
except AttributeError:
    pass

from ..tree.filtering import FilterList
from ..io import root_open
from ..plotting import Hist
from ..logger import multilogging
from .. import log; log = log[__name__]
from .student import Student

__all__ = [
    'Supervisor',
]

NCPUS = multiprocessing.cpu_count()


class QueueFeeder(Process):

    def __init__(self, connection, objects, queue, numclients, sentinel=None):

        Process.__init__(self)
        self.connection = connection
        self.objects = objects
        self.queue = queue
        self.numclients = numclients
        self.sentinel = sentinel

    def run(self):

        # ignore sigterm signal and let parent take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.queue.cancel_join_thread()
        self.objects = ([self.sentinel] * self.numclients) + \
            self.objects
        while self.objects:
            if self.connection.poll():
                print "queue feeder is shutting down..."
                break
            try:
                self.queue.put(self.objects[-1], 1)
                self.objects.pop()
            except Queue.Full:
                pass
        self.connection.close()
        print "queue feeder is closing the queue..."
        self.queue.close()
        print "queue feeder will now terminate"


class Supervisor(Process):

    def __init__(self,
                 student,
                 files,
                 outputname,
                 outputpath='.',
                 metadata=None,
                 nstudents=NCPUS,
                 connection=None,
                 gridmode=False,
                 queuemode=True,
                 nice=0,
                 name=None,
                 profile=False,
                 args=None,
                 **kwargs):

        super(Supervisor, self).__init__()
        self.process = student
        if isinstance(student, basestring):
            # remove .py extension if present
            student = os.path.splitext(student)[0]
            log.info("importing {0} ...".format(student))
            namespace = {}
            exec "from {0} import {1}".format(student, student) in namespace
            self.process = namespace[student]
        if not issubclass(self.process, Student):
            raise TypeError(
                "`{0}` must be a subclass of `Student`".format(student))
        if name is None:
            self.name = self.process.__name__
        else:
            self.name = name
        self.files = files[:]
        self.metadata = metadata
        self.outputname = '.'.join([self.name, outputname])
        self.outputpath = outputpath
        self.gridmode = gridmode
        self.nice = nice
        if self.gridmode:
            self.nstudents = 1
            queuemode = False
        else:
            if nstudents < 1:
                raise ValueError("`nstudents` must be at least 1")
            self.nstudents = min(nstudents, len(self.files))
        self.queuemode = queuemode
        self.kwargs = kwargs
        self.args = args
        self.connection = connection
        self.profile = profile
        self.student_outputs = list()
        self.process_table = dict()

    def run(self):

        # ignore sigterm signal and let parent take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        ROOT.gROOT.SetBatch()

        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener(os.path.join(
            self.outputpath,
            "supervisor-{0}-{1}.log".format(
                self.name, self.outputname)),
            self.logging_queue)
        self.listener.start()

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

        if self.queuemode:
            self.file_queue = multiprocessing.Queue(self.nstudents * 2)
            self.file_queue_feeder_conn, connection = multiprocessing.Pipe()
            self.file_queue_feeder = QueueFeeder(
                connection=connection,
                objects=self.files,
                queue=self.file_queue,
                numclients=self.nstudents,
                sentinel=None)

        self.output_queue = multiprocessing.Queue(-1)
        nfiles = len(self.files)
        log.info("Will run on {0:d} file{1}:".format(
            nfiles,
            's' if nfiles != 1 else ''))
        for filename in self.files:
            log.info(filename)
        sys.stdout.flush()
        self.hire_students()
        try:
            if self.supervise():
                self.publish()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])
        self.retire()

    def hire_students(self):
        """
        Create students for each block of files
        """
        log.info("defining students...")
        if self.queuemode:
            students = [
                self.process(
                    name=self.name,
                    files=self.file_queue,
                    output_queue=self.output_queue,
                    logging_queue=self.logging_queue,
                    gridmode=self.gridmode,
                    metadata=self.metadata,
                    profile=self.profile,
                    nice=self.nice,
                    args=self.args,
                    **self.kwargs
                ) for _ in xrange(self.nstudents)]
        else:
            # deal out files
            filesets = [[] for _ in xrange(self.nstudents)]
            while len(self.files) > 0:
                for fileset in filesets:
                    if len(self.files) > 0:
                        fileset.append(self.files.pop(0))
                    else:
                        break
            students = [
                self.process(
                    name=self.name,
                    files=fileset,
                    output_queue=self.output_queue,
                    logging_queue=self.logging_queue,
                    gridmode=self.gridmode,
                    metadata=self.metadata,
                    profile=self.profile,
                    nice=self.nice,
                    args=self.args,
                    **self.kwargs
                ) for fileset in filesets]
        for p in students:
            log.info("initialized student {0}".format(p))
            self.process_table[p.uuid] = p

    def supervise(self):
        """
        Supervise students until they have finished or until one fails
        """
        log.info("supervising students...")
        if self.queuemode:
            self.file_queue_feeder.start()
        for student in self.process_table.values():
            log.info("starting student {0}".format(student))
            student.start()
        while self.process_table:
            if self.connection is not None and self.connection.poll():
                msg = self.connection.recv()
                if msg is None:
                    log.info("received termination command")
                    return False
            while not self.output_queue.empty():
                id, output = self.output_queue.get()
                process = self.process_table[id]
                process.join()
                del self.process_table[id]
                if output is not None and process.exitcode == 0:
                    self.student_outputs.append(output)
                    log.info("student {0} finished successfully".format(
                        process))
                else:
                    log.error("student {0} failed".format(process))
                    return False
            time.sleep(1)
        return True

    def retire(self):
        """
        Shutdown the queues and terminate the remaining students
        """
        log.info("terminating...")
        for student in self.process_table.values():
            log.warning("terminating student {0}...".format(student))
            student.terminate()
        if self.queuemode:
            # tell queue feeder to quit
            log.debug("shutting down the file queue feeder...")
            self.file_queue_feeder_conn.send(None)
            log.debug("joining the file queue feeder process...")
            self.file_queue_feeder.join()
            log.debug("the file queue feeder process is terminated")
            self.file_queue_feeder_conn.close()
            log.debug("closing the file queue...")
            self.file_queue.close()
        log.debug("closing the output queue...")
        self.output_queue.close()
        log.debug("closing the logging queue...")
        self.logging_queue.put(None)
        self.listener.join()

    def publish(self):
        """
        Combine the output from all students
        """
        log.info("publishing output...")
        if len(self.student_outputs) > 0:
            outputs = []
            all_filters = []
            if self.profile:
                profiles = []
                for filters, output, profile in self.student_outputs:
                    all_filters.append(filters)
                    outputs.append(output)
                    profiles.append(profile)
                profile_output = StringIO.StringIO()
                profile_stats = pstats.Stats(profiles[0],
                                             stream=profile_output)
                for profile in profiles[1:]:
                    profile_stats.add(profile)
                profile_stats.sort_stats('cumulative').print_stats(50)
                print "\nProfiling Results: \n {0}".format(
                    profile_output.getvalue())
                for profile in profiles:
                    os.unlink(profile)
            else:
                for filters, output in self.student_outputs:
                    all_filters.append(filters)
                    outputs.append(output)

            write_cutflows = False
            if all_filters[0]:
                write_cutflows = True
                print("\n===== Cut-flow of filters for dataset "
                      "{0}: ====\n".format(self.outputname))

                merged_filters = dict([(
                    name,
                    reduce(
                        FilterList.merge,
                        [all_filters[i][name]
                            for i in xrange(len(all_filters))]))
                    for name in all_filters[0].keys()])

                for name, filterlist in merged_filters.items():
                    print "\n{0} cut-flow\n{1}\n".format(name, filterlist)

            outputname = os.path.join(
                self.outputpath, '{0}.root'.format(self.outputname))
            if os.path.exists(outputname):
                os.unlink(outputname)
            if len(outputs) == 1:
                shutil.move(outputs[0], outputname)
            else:
                self.process.merge(
                    outputs,
                    os.path.join(self.outputpath, self.outputname),
                    self.metadata)
                for output in outputs:
                    os.unlink(output)
            if write_cutflows:
                # write cut-flow in ROOT file as TH1
                with root_open(outputname, 'UPDATE'):
                    for name, filterlist in merged_filters.items():
                        cutflow = Hist(
                            len(filterlist) + 1, .5,
                            len(filterlist) + 1.5,
                            name="cutflow_{0}".format(name),
                            title="{0} cut-flow".format(name),
                            type='d')
                        cutflow[1].value = filterlist[0].total
                        cutflow.GetXaxis().SetBinLabel(1, "Total")
                        for i, filter in enumerate(filterlist):
                            cutflow[i + 2].value = filter.passing
                            cutflow.GetXaxis().SetBinLabel(i + 2, filter.name)
                        cutflow.Write()
                        # write count_func cutflow
                        for func_name in filterlist[0].count_funcs.keys():
                            cutflow = Hist(
                                len(filterlist) + 1, .5,
                                len(filterlist) + 1.5,
                                name="cutflow_{0}_{1}".format(
                                    name, func_name),
                                title="{0} {1} cut-flow".format(
                                    name, func_name),
                                type='d')
                            cutflow[1].value = filterlist[0].count_funcs_total[func_name]
                            cutflow.GetXaxis().SetBinLabel(1, "Total")
                            for i, filter in enumerate(filterlist):
                                # assume func_name in all filters
                                cutflow[i + 2].value = filter.count_funcs_passing[func_name]
                                cutflow.GetXaxis().SetBinLabel(i + 2, filter.name)
                            cutflow.Write()
