# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module provides context managers for filtering content on streams.
Probably the most common use is to completely silence output on stdout and/or
stderr with the `silence` and `silence_sout_serr` functions.

.. warning::
    There is the possibility that normal output may not be restored to the
    output stream and content may be unintentionally filtered. Only use these
    functions if you absolutely need them and beware of using them in large
    frameworks where debugging may be difficult if problems do occur.
"""
from contextlib import contextmanager, nested
from threading import Thread
import warnings

from tempfile import mkdtemp
from os.path import join as pjoin
from os import (dup, fdopen, open as osopen, O_NONBLOCK, O_RDONLY, remove,
                rmdir, mkfifo)
from fcntl import fcntl, F_GETFL, F_SETFL
from select import select
from sys import stdout, stderr

from ctypes import PyDLL, CDLL, c_void_p, c_char_p, py_object

__all__ = [
    'silence',
    'silence_sout_serr',
]

# suppress warning from python >=2.7
warnings.filterwarnings(
    'ignore',
    "With-statements now directly support multiple context managers",
    DeprecationWarning)

pyapi = PyDLL(None)
this_exe = CDLL(None)


def make_fn(what, res, *args):
    what.restype = res
    what.argtypes = args
    return what

FILE_p = c_void_p

PyFile_AsFile = make_fn(pyapi.PyFile_AsFile, FILE_p, py_object)
freopen = make_fn(this_exe.freopen, FILE_p, c_char_p, c_char_p, FILE_p)


@contextmanager
def fifo():
    """
    Create a fifo in a temporary place.
    """
    tmpdir = mkdtemp()
    filename = pjoin(tmpdir, 'myfifo')
    try:
        mkfifo(filename)
    except OSError, e:
        print >>stderr, "Failed to create FIFO: %s" % e
        raise
    else:
        yield filename
        remove(filename)
        rmdir(tmpdir)


def set_blocking(fd):
    """
    Set FD to be blocking
    """
    flags = fcntl(fd, F_GETFL)
    if flags & O_NONBLOCK:
        flags ^= O_NONBLOCK
    fcntl(fd, F_SETFL, flags)


def reader_thread_func(filename, filter_, real_stdout, filt_content):
    """
    Sit there, reading lines from the pipe `filename`, sending those for which
    `filter_()` returns False to `real_stdout`
    """
    with fdopen(osopen(filename, O_NONBLOCK | O_RDONLY)) as fd:
        while True:
            rlist, _, _ = select([fd], [], [])

            line = fd.readline()
            if not line:
                break

            elif filter_(line):
                filt_content.write(line)
            else:
                real_stdout.write(line)


@contextmanager
def threaded_file_reader(*args):
    """
    Operate a read_thread_func in another thread. Block with statement exit
    until the function completes.
    """
    reader_thread = Thread(target=reader_thread_func, args=args)
    reader_thread.start()
    try:
        yield
    finally:
        reader_thread.join()


@contextmanager
def silence(filter_=lambda line: True, file_=stdout):
    """
    Prevent lines matching `filter_` ending up on `file_` (defaults to stdout).
    By default filter all lines.
    """
    if not filter_:
        yield
        return

    saved_stdout = dup(file_.fileno())
    stdout_file = PyFile_AsFile(file_)

    from cStringIO import StringIO
    filt_content = StringIO()

    with nested(fdopen(saved_stdout, "w"), fifo()) as (real_stdout, filename):
        tfr = threaded_file_reader(filename, filter_, real_stdout, filt_content)
        with tfr:
            # Redirect stdout to pipe
            freopen(filename, "w", stdout_file)
            try:
                yield filt_content
            finally:
                # Redirect stdout back to its original place
                freopen("/dev/fd/%i" % saved_stdout, "w", stdout_file)


@contextmanager
def silence_sout_serr(filter_=lambda line: True):
    """
    Prevent lines matching `filter_` ending up on stdout or stderr.
    By default filter all lines.
    """
    from sys import stdout, stderr
    with nested(silence(filter_, stdout),
                silence(filter_, stderr)) as (so, se):
        yield so, se
