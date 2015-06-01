# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.defaults import use_rootpy_handler, use_rootpy_magic

if not use_rootpy_handler or not use_rootpy_magic:
    from nose.plugins.skip import SkipTest
    raise SkipTest()

import logging
import sys

from nose.tools import raises

import ROOT

import rootpy
import rootpy.logger.magic as M

from rootpy import ROOTError
from .logcheck import EnsureLogContains

M.DANGER.enabled = True

NONEXISTENT_FILE = "this-file-should-never-exist-7b078562896325fa8007a0eb0.root"

#rootpy.log["/ROOT.rootpy"].show_stack(".*tracing.*")


@EnsureLogContains("WARNING", "^This is a test message$")
def test_logging_root_messages():
    ROOT.Warning("rootpy.logger.tests", "This is a test message")


@raises(ROOTError)
def test_root_error():
    ROOT.Error("rootpy.logger.tests", "This is a test exception")


@raises(ROOTError)
def test_nonexistent_file():
    ROOT.TFile(NONEXISTENT_FILE)


@raises(ROOTError)
def test_error_finally():
    try:
        ROOT.Error("test", "finally")
    finally:
        test = 1


@raises(ROOTError)
def test_gdebug_finally():
    ROOT.gDebug = 1
    try:
        ROOT.Error("test", "finally [rootpy.ALWAYSABORT]")
    finally:
        ROOT.gDebug = 0


def test_nonexistent_file_redux():
    try:
        ROOT.TFile(NONEXISTENT_FILE)
    except ROOTError as e:
        assert e.location == "TFile::TFile"
        assert e.level == 3000
        assert NONEXISTENT_FILE in e.msg
        assert "does not exist" in e.msg
    else:
        assert False, "Should have thrown"

# The following tests ensure that things work as expected with different
# constructs

@raises(ROOTError)
def test_nonexistent_file_redux_part_2():
    if True:
        ROOT.TFile(NONEXISTENT_FILE)


@raises(ROOTError)
def test_nonexistent_file_redux_part_3_the_loopening():
    for i in range(10):
        ROOT.TFile(NONEXISTENT_FILE)


@raises(ROOTError)
def test_nonexistent_file_redux_part_4_the_withinating():
    class Context(object):
        def __enter__(*args): pass
        def __exit__(*args): pass

    with Context():
        ROOT.TFile(NONEXISTENT_FILE)


def test_correct_bytecode_functioning():
    # This test ensures that we don't break opcodes which follow exceptions

    fail = True
    class Continued:
        success = False

    def try_fail():
        if fail:
            ROOT.Error("rooypy.logger.tests", "TEST")
        Continued.success = True

    if sys.version_info[0] < 3:
        orig_code_bytes = [ord(i) for i in try_fail.func_code.co_code]
    else:
        orig_code_bytes = try_fail.__code__.co_code

    #import dis
    #dis.dis(try_fail)

    try:
        try_fail()
    except ROOTError:
        pass
    else:
        assert False, "Should have thrown"

    #print "#"*80
    #dis.dis(try_fail)
    if sys.version_info[0] < 3:
        new_code_bytes = [ord(i) for i in try_fail.func_code.co_code]
    else:
        new_code_bytes = try_fail.__code__.co_code

    assert orig_code_bytes == new_code_bytes

    fail = False
    try_fail()

    assert Continued.success


def test_tracing_is_broken():
    def mytrace(*args):
        pass

    orig_trace = sys.gettrace()
    sys.settrace(mytrace)

    try:
        ROOT.Error("rootpy.logger.tests", "Test tracing OK")
    except ROOTError:
        pass
    else:
        assert False, "Should have thrown"

    should_be_mytrace = sys.gettrace()
    sys.settrace(orig_trace)

    assert should_be_mytrace != mytrace, "Tracing is fixed?! Awesome. Now fix the test."
