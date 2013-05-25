# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.io import TemporaryFile, DoesNotExist
from rootpy.plotting import Hist
from rootpy.testdata import get_file
from nose.tools import assert_raises, assert_equals
import os


def test_tempfile():

    with TemporaryFile():
        Hist(1, 0, 1, name='test').write()


def test_file():

    f = TemporaryFile()
    assert_raises(DoesNotExist, f.Get, 'blah')
    hist = Hist(1, 0, 1, name='test')
    hist.Write()
    hist2 = f.test
    assert hist2.__class__ == hist.__class__
    os.unlink(f.GetName())


def test_file_get():

    f = get_file()
    d = f.Get('means', rootpy=False)
    assert_equals(d.__class__.__name__, 'TDirectoryFile')
    d = f.Get('means')
    assert_equals(d.__class__.__name__, 'Directory')
    h = f.Get('means/hist1', rootpy=False)
    assert_equals(h.__class__.__name__, 'TH1F')
    h = f.Get('means/hist1')
    assert_equals(h.__class__.__name__, 'Hist')


def test_file_item():

    with TemporaryFile() as f:
        h = Hist(1, 0, 1, name='test')
        f['myhist'] = h
        f.myhist
        assert_equals(f['myhist'].name, 'test')


def test_file_attr():

    with TemporaryFile() as f:
        h = Hist(1, 0, 1, name='test')
        f.myhist = h
        f.Get('myhist')
        assert_equals(f.myhist.name, 'test')
        f.something = 123


if __name__ == "__main__":
    import nose
    nose.runmodule()
