# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
# multiprocessing uses the exceptions from the Queue module
import Queue
from ..tree.filtering import FilterList
from ..io import root_open
from ..plotting import Hist

from ..logger import multilogging
from .. import log; log = log[__name__]
import logging
try:
    logging.captureWarnings(True)
except AttributeError:
    pass

import traceback
import signal
from .student import Student
import pstats
import cStringIO as StringIO
import shutil


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

        Process.__init__(self)

        self.process = student
        if isinstance(student, basestring):
            # remove .py extension if present
            student = os.path.splitext(student)[0]
            log.info("importing %s ..." % student)
            exec "from %s import %s" % (student, student)
            self.process = eval(student)
        if not issubclass(self.process, Student):
            raise TypeError("%s must be a subclass of Student" % student)

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
                raise ValueError('``nstudents`` must be at least 1')
            self.nstudents = min(nstudents, len(self.files))
        self.queuemode = queuemode
        self.student_outputs = []
        self.kwargs = kwargs
        self.args = args
        self.connection = connection
        self.profile = profile

    def run(self):

        # ignore sigterm signal and let parent take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        ROOT.gROOT.SetBatch()

        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener(os.path.join(self.outputpath,
            "supervisor-%s-%s.log" %
            (self.name, self.outputname)), self.logging_queue)
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
        try:
            log.info("Will run on %i file(s):" % len(self.files))
            for filename in self.files:
                log.info("%s" % filename)
            sys.stdout.flush()
            self.hire_students()
            self.supervise()
            self.publish()
        except:
            print sys.exc_info()
            traceback.print_tb(sys.exc_info()[2])

        if self.queuemode:
            self.file_queue.close()
        self.output_queue.close()
        self.logging_queue.put(None)
        self.listener.join()
        log.info("Done")

    def hire_students(self):

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
        self.process_table = dict([(p.uuid, p) for p in students])

    def supervise(self):

        if self.queuemode:
            self.file_queue_feeder.start()
        for student in self.process_table.values():
            student.start()
        while self.process_table:
            if self.connection is not None:
                if self.connection.poll():
                    msg = self.connection.recv()
                    if msg is None:
                        self.retire()
                        return
            while not self.output_queue.empty():
                id, output = self.output_queue.get()
                process = self.process_table[id]
                process.join()
                del self.process_table[id]
                if output is not None and process.exitcode == 0:
                    self.student_outputs.append(output)
                else:
                    log.error("a student has failed")
                    self.retire()
                    return

            time.sleep(1)

    def retire(self):

        log.info("will now terminate...")
        if self.queuemode:
            # tell queue feeder to quit
            log.info("shutting down file queue...")
            self.file_queue_feeder_conn.send(None)
            log.info("joining queue feeder...")
            self.file_queue_feeder.join()
            log.info("queue feeder is terminated")
            self.file_queue_feeder_conn.close()
        log.info("terminating students...")
        for student in self.process_table.values():
            student.terminate()

    def publish(self):

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
                print "\nProfiling Results: \n %s" % profile_output.getvalue()
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
                      "%s: ====\n" % self.outputname)

                merged_filters = dict([(name, reduce(FilterList.merge,
                                  [all_filters[i][name] for i in
                                      xrange(len(all_filters))])) for name in
                                      all_filters[0].keys()])

                for name, filterlist in merged_filters.items():
                    print "\n%s cut-flow\n%s\n" % (name, filterlist)

            outputname = os.path.join(
                self.outputpath, '%s.root' % self.outputname)
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
                                name="cutflow_%s" % name,
                                title="%s cut-flow" % name,
                                type='d')
                        cutflow[0] = filterlist[0].total
                        cutflow.GetXaxis().SetBinLabel(1, "Total")
                        for i, filter in enumerate(filterlist):
                            cutflow[i + 1] = filter.passing
                            cutflow.GetXaxis().SetBinLabel(i + 2, filter.name)
                        cutflow.Write()
                        # write count_func cutflow
                        for func_name in filterlist[0].count_funcs.keys():
                            cutflow = Hist(
                                    len(filterlist) + 1, .5,
                                    len(filterlist) + 1.5,
                                    name="cutflow_%s_%s" % (name, func_name),
                                    title="%s %s cut-flow" % (name, func_name),
                                    type='d')
                            cutflow[0] = filterlist[0].count_funcs_total[func_name]
                            cutflow.GetXaxis().SetBinLabel(1, "Total")
                            for i, filter in enumerate(filterlist):
                                # assume func_name in all filters
                                cutflow[i + 1] = filter.count_funcs_passing[func_name]
                                cutflow.GetXaxis().SetBinLabel(i + 2, filter.name)
                            cutflow.Write()
