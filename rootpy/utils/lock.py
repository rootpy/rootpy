# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os
import stat
import time
import platform
from contextlib import contextmanager

from ..extern.lockfile import LockFile, LockTimeout
from . import log; log = log[__name__]

__all__ = [
    'lock',
]


@contextmanager
def lock(path, poll_interval=5, max_age=60):
    """
    Aquire a file lock in a thread-safe manner that also reaps stale locks
    possibly left behind by processes that crashed hard.
    """
    if poll_interval >= max_age:
        raise ValueError("`poll_interval` must be less than `max_age`")
    proc = '{0:d}@{1}'.format(os.getpid(), platform.node())
    lock = LockFile(path)
    log.debug("{0} attempting to lock {1}".format(proc, path))
    while not lock.i_am_locking():
        if lock.is_locked():
            # check age of the lock file
            age = time.time() - os.stat(lock.lock_file)[stat.ST_MTIME]
            # break the lock if too old (considered stale)
            if age > max_age:
                log.warning(
                    "{0} breaking exiting lock on {1} "
                    "that is {2:d} seconds old".format(
                        proc, path, int(age)))
                lock.break_lock()
        try:
            log.debug(
                "{0} waiting for {1:d} seconds "
                "for lock on {2} to be released".format(
                    proc, poll_interval, path))
            lock.acquire(timeout=poll_interval)
        except LockTimeout:
            pass
    log.debug("{0} locked {1}".format(proc, path))
    yield lock
    lock.release()
    log.debug("{0} released lock on {1}".format(proc, path))
