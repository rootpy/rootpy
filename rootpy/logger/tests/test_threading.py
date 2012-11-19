from __future__ import division

import itertools
import os
import threading
import time

from random import random

import rootpy; log = rootpy.log["rootpy.logger.test.threading"]
rootpy.logger.magic.DANGER.enabled = True

import ROOT

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

#@EnsureLogContains("ERROR", "ALWAYSABORT")
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
        for i in range(100):
            t = threading.Thread(target=randomfatal, args=(should_exit,))
            t.start()
            threads.append(t)

        time.sleep(length)

        should_exit.set()
        for t in threads:
            t.join()

    finally:
        #sup_logger.setLevel(old_level)
        pass


    tot = total.next()-1
    fatals = number_of_fatals.next()-1
    log.debug("Success raising exceptions: total: {0} (fatals {1:%})".format(tot, fatals / tot))