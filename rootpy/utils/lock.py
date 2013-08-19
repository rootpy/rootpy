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
    if max_age < 30:
        raise ValueError("`max_age` must be at least 30 seconds")
    if poll_interval < 1:
        raise ValueError("`poll_interval` must be at least 1 second")
    if poll_interval >= max_age:
        raise ValueError("`poll_interval` must be less than `max_age`")
    proc = '{0:d}@{1}'.format(os.getpid(), platform.node())
    lock = LockFile(path)
    log.debug("{0} attempting to lock {1}".format(proc, path))
    while not lock.i_am_locking():
        if lock.is_locked():
            # Protect against race condition
            try:
                # Check age of the lock file
                age = time.time() - os.stat(lock.lock_file)[stat.ST_MTIME]
                # Break the lock if too old (considered stale)
                if age > max_age:
                    lock.break_lock()
                    # What if lock was released and reacquired in the meantime?
                    # We don't want to break a fresh lock!
                    # If a lock is stale then we may have many threads
                    # attempting to break it here at the "same time".
                    # Avoid the possibility of some thread trying to break the
                    # lock after it has already been broken and after the first
                    # other thread attempting to acquire the lock by sleeping
                    # for 0.5 seconds below.
                    log.warning(
                        "{0} broke lock on {1} "
                        "that is {2:d} seconds old".format(
                            proc, path, int(age)))
            except OSError:
                # Lock was released just now
                # os.path.exists(lock.lock_file) is False
                # OSError may be raised by os.stat() or lock.break_lock() above
                pass
        time.sleep(0.5)
        try:
            log.debug(
                "{0} waiting for {1:d} seconds "
                "for lock on {2} to be released".format(
                    proc, poll_interval, path))
            # Use float() here since acquire sleeps for timeout/10
            lock.acquire(timeout=float(poll_interval))
        except LockTimeout:
            pass
    log.debug("{0} locked {1}".format(proc, path))
    yield lock
    lock.release()
    log.debug("{0} released lock on {1}".format(proc, path))
