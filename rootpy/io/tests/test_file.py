# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.context import invisible_canvas
from rootpy.io import TemporaryFile, DoesNotExist
from rootpy.plotting import Hist
from rootpy.testdata import get_file

from nose.tools import assert_raises, assert_equals

import gc
import os

import ROOT as R


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
        f.mkdir('hello')
        f.hello.something = h
        assert_equals(f['hello/something'].name, 'test')

def test_no_dangling_files():
    
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
