# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.context import invisible_canvas
from rootpy.io import TemporaryFile, DoesNotExist, MemFile, File, Directory
from rootpy.io import root_open
from rootpy.plotting import Hist
from rootpy.testdata import get_file
from rootpy import ROOT

from nose.tools import assert_raises, assert_equal, assert_true

import gc
import os

import ROOT as R


def test_tempfile():
    with TemporaryFile() as f:
        assert_equal(os.path.isfile(f.GetName()), True)
        assert_raises(DoesNotExist, f.Get, 'blah')
        hist = Hist(1, 0, 1, name='test')
        hist.Write()
        hist2 = f.test
        assert_equal(hist2.__class__, hist.__class__)
    assert_equal(os.path.isfile(f.GetName()), False)


def test_memfile():
    with MemFile() as f:
        hist = Hist(1, 0, 1, name='test')
        hist.Write()
        assert_equal(f['test'], hist)


def test_file_open():
    fname = 'test_file_open.root'
    with File.open(fname, 'w'):
        pass
    with root_open(fname, 'r'):
        pass
    with root_open(fname):
        pass
    os.unlink(fname)


def test_context():
    with MemFile() as a:
        assert_equal(ROOT.gDirectory.func(), a)
        with MemFile() as b:
            d = Directory('test')
            with d:
                assert_equal(ROOT.gDirectory.func(), d)
            assert_equal(ROOT.gDirectory.func(), b)
        assert_equal(ROOT.gDirectory.func(), a)

    # test out of order
    f1 = MemFile()
    f2 = MemFile()
    with f1:
        assert_equal(ROOT.gDirectory.func(), f1)
    assert_equal(ROOT.gDirectory.func(), f2)
    f1.Close()
    f2.Close()

    d = Directory('test')
    d.cd()

    # test without with statement
    f1 = MemFile()
    f2 = TemporaryFile()
    assert_equal(ROOT.gDirectory.func(), f2)
    f2.Close()
    assert_equal(ROOT.gDirectory.func(), f1)
    f1.Close()


def test_file_get():
    with get_file() as f:
        d = f.Get('means', rootpy=False)
        assert_equal(d.__class__.__name__, 'TDirectoryFile')
        d = f.Get('means')
        assert_equal(d.__class__.__name__, 'Directory')
        h = f.Get('means/hist1', rootpy=False)
        assert_equal(h.__class__.__name__, 'TH1F')
        h = f.Get('means/hist1')
        assert_equal(h.__class__.__name__, 'Hist')


def test_file_item():
    with TemporaryFile() as f:
        h = Hist(1, 0, 1, name='test')
        f['myhist'] = h
        f.myhist
        assert_equal(f['myhist'].name, 'test')


def test_file_attr():
    with TemporaryFile() as f:
        h = Hist(1, 0, 1, name='test')
        f.myhist = h
        f.Get('myhist')
        assert_equal(f.myhist.name, 'test')
        f.something = 123
        f.mkdir('hello')
        f.hello.something = h
        assert_equal(f['hello/something'].name, 'test')


def test_file_contains():
    with TemporaryFile() as f:
        assert_equal('some/thing' in f, False)
        rdir = f.mkdir('some')
        thing = Hist(10, 0, 1, name='thing')
        rdir.thing = thing
        assert_true('some/thing' in f)
        assert_true('thing' in rdir)
        f.mkdir('a/b/c', recurse=True)
        assert_true('a/b/c' in f)


def test_no_dangling_files():

    def foo():
        f = MemFile()

    foo()

    g = root_open('test_no_dangling_files.root', 'recreate')
    os.unlink('test_no_dangling_files.root')
    del g

    gc.collect()
    assert list(R.gROOT.GetListOfFiles()) == [], "There exist open ROOT files when there should not be"


def test_keepalive():
    gc.collect()
    assert list(R.gROOT.GetListOfFiles()) == [], "There exist open ROOT files when there should not be"

    # Ordinarily this would lead to h with a value of `None`, since the file
    # gets garbage collected. However, File.Get uses keepalive to prevent this.
    # The purpose of this test is to ensure that everything is working as
    # expected.
    h = get_file().Get("means/hist1")
    gc.collect()
    assert h, "hist1 is not being kept alive"
    assert list(R.gROOT.GetListOfFiles()) != [], "Expected an open ROOT file.."

    h = None
    gc.collect()
    assert list(R.gROOT.GetListOfFiles()) == [], "There exist open ROOT files when there should not be"


def test_keepalive_canvas():

    gc.collect()
    assert list(R.gROOT.GetListOfFiles()) == [], "There exist open ROOT files when there should not be"

    with invisible_canvas() as c:
        get_file().Get("means/hist1").Draw()
        gc.collect()
        assert list(R.gROOT.GetListOfFiles()) != [], "Expected an open ROOT file.."

    gc.collect()
    assert list(R.gROOT.GetListOfFiles()) == [], "There exist open ROOT files when there should not be"


if __name__ == "__main__":
    import nose
    nose.runmodule()
