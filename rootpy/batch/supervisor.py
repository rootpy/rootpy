import ROOT
import time
import os
import sys
import multiprocessing
from multiprocessing import Process
from operator import add, itemgetter
import uuid
from ..tree.filtering import *
from .. import common
from . import multilogging
import logging
import traceback
import signal
from .student import Student
import cPickle as pickle
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
            except multiprocessing.queues.Queue.Full:
                pass
        self.connection.close()
        print "queue feeder is closing the queue..."
        self.queue.close()
        print "queue feeder will now terminate"


class Supervisor(Process):

    def __init__(self, student, outputname,
                 files,
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
            print "importing %s..." % student
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
        self.gridmode = gridmode
        self.nice = nice
        if self.gridmode:
            self.nstudents = 1
            queuemode = False
        else:
            self.nstudents = min(nstudents, len(self.files))
        self.queuemode = queuemode
        self.student_outputs = []
        self.kwargs = kwargs
        self.logger = None
        self.args = args
        self.connection = connection
        self.profile = profile

    def run(self):

        # ignore sigterm signal and let parent take care of this
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        ROOT.gROOT.SetBatch()

        # logging
        self.logging_queue = multiprocessing.Queue(-1)
        self.listener = multilogging.Listener("supervisor-%s-%s.log" % \
            (self.name, self.outputname), self.logging_queue)
        self.listener.start()

        h = multilogging.QueueHandler(self.logging_queue)
        self.logger = logging.getLogger('Supervisor')
        self.logger.addHandler(h)
        self.logger.setLevel(logging.DEBUG)

        if not self.gridmode:
            sys.stdout = multilogging.stdout(self.logger)
            sys.stderr = multilogging.stderr(self.logger)

        if self.queuemode:
            self.file_queue = multiprocessing.Queue(self.nstudents * 2)
            self.file_queue_feeder_conn, connection = multiprocessing.Pipe()
            self.file_queue_feeder = QueueFeeder(connection=connection,
                                                 objects=self.files,
                                                 queue=self.file_queue,
                                                 numclients=self.nstudents,
                                                 sentinel=None)

        self.output_queue = multiprocessing.Queue(-1)
        try:
            print "Will run on %i file(s):" % len(self.files)
            for filename in self.files:
                print "%s" % filename
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
        print "Done"

    def hire_students(self):

        if self.queuemode:
            students = [
                self.process(
                    name = self.name,
                    files = self.file_queue,
                    output_queue = self.output_queue,
                    logging_queue = self.logging_queue,
                    gridmode = self.gridmode,
                    metadata = self.metadata,
                    profile = self.profile,
                    nice = self.nice,
                    args = self.args,
                    **self.kwargs
                ) for i in xrange(self.nstudents) ]
        else:
            # deal out files
            filesets = [[] for i in xrange(self.nstudents)]
            while len(self.files) > 0:
                for fileset in filesets:
                    if len(self.files) > 0:
                        fileset.append(self.files.pop(0))
                    else:
                        break
            students = [
                self.process(
                    name = self.name,
                    files = fileset,
                    output_queue = self.output_queue,
                    logging_queue = self.logging_queue,
                    gridmode = self.gridmode,
                    metadata = self.metadata,
                    profile = self.profile,
                    nice = self.nice,
                    args = self.args,
                    **self.kwargs
                ) for fileset in filesets ]
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
                        print "will now terminate..."
                        if self.queuemode:
                            # tell queue feeder to quit
                            print "shutting down file queue..."
                            self.file_queue_feeder_conn.send(None)
                            print "joining queue feeder..."
                            self.file_queue_feeder.join()
                            print "queue feeder is terminated"
                            self.file_queue_feeder_conn.close()
                        print "terminating students..."
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

    def publish(self, merge=True):

        if len(self.student_outputs) > 0:
            outputs = []
            event_filters = []
            object_filters = []
            if self.profile:
                profiles = []
                for event_filter, object_filter, output, profile in self.student_outputs:
                    event_filters.append(event_filter)
                    object_filters.append(object_filter)
                    outputs.append(output)
                    profiles.append(profile)
                profile_output = StringIO.StringIO()
                profile_stats = pstats.Stats(profiles[0], stream=profile_output)
                for profile in profiles[1:]:
                    profile_stats.add(profile)
                profile_stats.sort_stats('cumulative').print_stats(50)
                print "\nProfiling Results: \n %s" % profile_output.getvalue()
                for profile in profiles:
                    os.unlink(profile)
            else:
                for event_filter, object_filter, output in self.student_outputs:
                    event_filters.append(event_filter)
                    object_filters.append(object_filter)
                    outputs.append(output)

            print "\n===== Cut-flow of event filters for dataset %s: ====\n"% self.outputname
            totalEvents = 0
            combinedEventFilterlist = reduce(FilterList.merge, event_filters)
            combinedObjectFilterlist = reduce(FilterList.merge, object_filters)
            totalEvents = combinedEventFilterlist.total
            print "Event Filters:\n%s"% combinedEventFilterlist
            print "Object Filters:\n%s"% combinedObjectFilterlist

            with open("cutflow.p",'w') as pfile:
                pickle.dump({"event": combinedEventFilterlist.basic(),
                             "object": combinedObjectFilterlist.basic()}, pfile)

            if merge:
                outputname = '%s.root' % self.outputname
                if os.path.exists(outputname):
                    os.unlink(outputname)
                if len(outputs) == 1:
                    shutil.move(outputs[0], outputname)
                else:
                    self.process.merge(outputs, self.outputname, self.metadata)
                for output in outputs:
                    os.unlink(output)
