# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module provides context managers for silencing output from external
compiled libraries on stdout, stderr, or both. Probably the most common use
is to completely silence output on stdout and/or stderr with the
`silence_sout_serr` function.

.. warning::
    There is the possibility that normal output may not be restored to the
    output stream and content may be unintentionally silenced. Only use these
    functions if you absolutely need them and beware of using them in large
    frameworks where debugging may be difficult if problems do occur.

"""
from contextlib import contextmanager
import os
import sys
import threading
LOCK = threading.RLock()

__all__ = [
    'silence_sout',
    'silence_serr',
    'silence_sout_serr',
]


@contextmanager
def silence_sout():
    LOCK.acquire()
    sys.__stdout__.flush()
    origstdout = sys.__stdout__
    oldstdout_fno = os.dup(sys.__stdout__.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.__stdout__ = os.fdopen(newstdout, 'w')
    try:
        yield
    finally:
        sys.__stdout__ = origstdout
        sys.__stdout__.flush()
        os.dup2(oldstdout_fno, 1)
        LOCK.release()


@contextmanager
def silence_serr():
    LOCK.acquire()
    sys.__stderr__.flush()
    origstderr = sys.__stderr__
    oldstderr_fno = os.dup(sys.__stderr__.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.__stderr__ = os.fdopen(newstderr, 'w')
    try:
        yield
    finally:
        sys.__stderr__ = origstderr
        sys.__stderr__.flush()
        os.dup2(oldstderr_fno, 2)
        LOCK.release()


@contextmanager
def silence_sout_serr():
    with silence_sout():
        with silence_serr():
            yield
