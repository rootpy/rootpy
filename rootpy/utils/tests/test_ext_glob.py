import os
from glob import glob as py_glob
from nose.tools import assert_equal, assert_true
from nose.plugins.attrib import attr
from rootpy.utils.ext_glob import glob as ext_glob


# It's conceivable that these parameters might be worth overloading / changing
# from a config file or command line option, particulary the remote_directory
this_directory = os.path.dirname(os.path.abspath(__file__))
remote_directory = "root://eospublic.cern.ch//eos/report/eospublic/2015/11"


def test_local_glob_none():
    filename = os.path.join(this_directory, "test_ext_glob.py")
    assert_equal([filename], ext_glob(filename))


def test_local_glob_basename():
    filename = os.path.join(this_directory, "*.py")
    assert_equal(ext_glob(filename), py_glob(filename))


def test_local_glob_dirname():
    filename = os.path.join(os.path.dirname(this_directory), "*")
    assert_equal(ext_glob(filename), py_glob(filename))


def test_local_glob_both():
    filename = os.path.join(os.path.dirname(this_directory), "*", "*.py")
    assert_equal(ext_glob(filename), py_glob(filename))


def test_xrootd_glob_none():
    filename = remote_directory
    assert_equal(ext_glob(filename),[filename])


@attr('network')
def test_xrootd_glob_single():
    filename = "/".join([remote_directory, '*'])
    assert_true(len(ext_glob(filename)) >= 1)


@attr('network')
def test_xrootd_glob_multiple():
    filename = os.path.dirname(remote_directory)
    filename = "/".join([filename, '*', '*'])
    assert_true(len(ext_glob(filename)) >= 2)
