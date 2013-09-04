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

__all__ = [
    'silence_sout',
    'silence_serr',
    'silence_sout_serr',
]


@contextmanager
def silence_sout():

    sys.stdout.flush()
    origstdout = sys.stdout
    oldstdout_fno = os.dup(sys.stdout.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')
    try:
        yield
    finally:
        sys.stdout = origstdout
        sys.stdout.flush()
        os.dup2(oldstdout_fno, 1)


@contextmanager
def silence_serr():

    sys.stderr.flush()
    origstderr = sys.stderr
    oldstderr_fno = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stderr = os.fdopen(newstderr, 'w')
    try:
        yield
    finally:
        sys.stderr = origstderr
        sys.stderr.flush()
        os.dup2(oldstderr_fno, 2)


@contextmanager
def silence_sout_serr():

    with silence_sout():
        with silence_serr():
            yield
