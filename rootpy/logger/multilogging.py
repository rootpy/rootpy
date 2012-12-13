# rootpy license excluded in this source file

# Copyright (C) 2010 Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

"""
How to use logging with multiprocessing
---------------------------------------

The basic strategy is to set up a listener process which can have any logging
configuration you want - in this example, writing to rotated log files. Because
only the listener process writes to the log files, you don't have file
corruption caused by multiple processes trying to write to the file.

The listener process is initialised with a queue, and waits for logging events
(LogRecords) to appear in the queue. When they do, they are processed according
to whatever logging configuration is in effect for the listener process.

Other processes can delegate all logging to the listener process. They can have
a much simpler logging configuration: just one handler, a QueueHandler, needs
to be added to the root logger. Other loggers in the configuration can be set
up with levels and filters to achieve the logging verbosity you need.

A QueueHandler processes events by sending them to the multiprocessing queue
that it's initialised with.
"""
import logging
import logging.handlers
import multiprocessing


class stdlog(object):

    def __init__(self, logger):
        self.logger = logger

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def write(self, s):
        raise NotImplementedError


class stdout(stdlog):

    def write(self, s):
        s = s.strip()
        if s:
            self.logger.info(s)


class stderr(stdlog):

    def write(self, s):
        s = s.strip()
        if s:
            self.logger.error(s)


class QueueHandler(logging.Handler):
    """ This is a logging handler which sends events to a multiprocessing queue.
    The plan is to add it to Python 3.2, but this can be copy pasted into
    user code for use with earlier Python versions.
    """
    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        # fix TypeError: super() argument 1 must be type, not classobj
        # in Python 2.6, don't use super()
        # (in 2.6 the logging.Handler is an old style class)
        logging.Handler.__init__(self)
        self.queue = queue

    def emit(self, record):
        """ Emit a record.
        Writes the LogRecord to the queue.
        """
        try:
            if record.exc_info:
                # just to get traceback text into record.exc_text
                dummy = self.format(record)
                # not needed any more
                record.exc_info = None
            self.queue.put_nowait(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class Listener(multiprocessing.Process):
    """
    Because you'll want to define the logging configurations for listener and
    workers, the listener and worker process functions take a configurer
    parameter which is a callable for configuring logging for that process.
    These functions are also passed the queue, which they use for communication.

    In practice, you can configure the listener however you want, but note that
    in this simple example, the listener does not apply level or filter logic to
    received records. In practice, you would probably want to do ths logic in
    the worker processes, to avoid sending events which would be filtered out
    between processes.

    The size of the rotated files is made small so you can see the results
    easily.

    This is the listener process top-level loop: wait for logging events
    (LogRecords) on the queue and handle them, quit when you get a None for a
    LogRecord.
    """
    def __init__(self, name, queue, capacity = 1, *args, **kwargs):

        super(Listener, self).__init__(*args, **kwargs)
        self.capacity = capacity
        self.queue = queue
        self.name = name

    def run(self):

        root = logging.getLogger()
        h = logging.handlers.RotatingFileHandler(
                self.name,
                mode='w')
        memoryHandler = logging.handlers.MemoryHandler(
                capacity=self.capacity,
                target=h)
        f = logging.Formatter(
                '%(asctime)s %(processName)-10s '
                '%(name)s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        root.addHandler(memoryHandler)

        while True:
            try:
                record = self.queue.get()
                # We send this as a sentinel to tell the listener to quit.
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                # No level or filter logic applied - just do it!
                logger.handle(record)
            except (KeyboardInterrupt, SystemExit):
                try:
                    memoryHandler.close()
                except:
                    pass
                raise
            except:
                import sys, traceback
                print >> sys.stderr, 'multilogging problem:'
                traceback.print_exc(file=sys.stderr)

        memoryHandler.close()
