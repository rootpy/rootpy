# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import division

from rootpy.defaults import use_rootpy_handler, use_rootpy_magic

if not use_rootpy_handler or not use_rootpy_magic:
    from nose.plugins.skip import SkipTest
    raise SkipTest()

import itertools
import os
import os.path
import platform
import resource
import thread
import threading
import time

from math import ceil
from random import random

import ROOT

import rootpy; log = rootpy.log["rootpy.logger.test.threading"]
rootpy.logger.magic.DANGER.enabled = True

from .logcheck import EnsureLogContains

def optional_fatal(abort=True):
    msg = "[rootpy.ALWAYSABORT]" if abort else "[rootpy.NEVERABORT]"
    ROOT.Error("rootpy.logger.test", msg)

f = optional_fatal
optional_fatal._bytecode = lambda: map(ord, f.func_code.co_code)
optional_fatal._ORIG_BYTECODE = optional_fatal._bytecode()
optional_fatal._unmodified = lambda: f._bytecode() == f._ORIG_BYTECODE

def optional_fatal_bytecode_check():
    assert optional_fatal._unmodified(), (
        "Detected modified bytecode. This should never happen.")

number_of_fatals = itertools.count()
total = itertools.count()

def maybe_fatal():
    try:
        # Throw exceptions 80% of the time
        optional_fatal(random() < 0.8)
    except rootpy.ROOTError:
        number_of_fatals.next()
    finally:
        total.next()
        optional_fatal_bytecode_check()

def randomfatal(should_exit):
    while not should_exit.is_set():
        maybe_fatal()

def spareprocs():
    """
    Compute the maximum number of threads we can start up according to ulimit
    """
    if not os.path.exists("/proc"):
        # Return a decent small value, we just want it to run, more grindy tests
        # can take place on other machines.
        return 10

    nmax, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    me = os.geteuid()
    return nmax - sum(1 for p in os.listdir("/proc")
                       if p.isdigit() and os.stat("/proc/" + p).st_uid == me)

def test_multithread_exceptions():
    should_exit = threading.Event()

    sup_logger = log["/ROOT.rootpy.logger.test"]
    old_level = sup_logger.level
    # Suppress test warnings
    sup_logger.setLevel(log.CRITICAL)

    # Run for 1/4 second or 10s if LONG_TESTS is in the environment
    length = float(os.environ.get("TEST_TIME", 0.25))

    try:
        threads = []
        for i in range(min(100, int(ceil(spareprocs()*0.8)))):
            t = threading.Thread(target=randomfatal, args=(should_exit,))
            try:
                t.start()
                threads.append(t)
            except thread.error:
                log.warning("Unable to start thread")
                break

        assert threads, "Didn't manage to start any threads!"

        time.sleep(length)

        should_exit.set()
        for t in threads:
            t.join()

    finally:
        sup_logger.setLevel(old_level)

    tot = total.next()-1
    fatals = number_of_fatals.next()-1
    fmt = "Success raising exceptions in {0} threads: total: {1} (fatals {2:%})"
    log.debug(fmt.format(len(threads), tot, fatals / tot))
