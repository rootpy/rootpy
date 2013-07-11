# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.io import TemporaryFile
from rootpy.io.pickler import load, dump
from rootpy.plotting import Hist
import random
from nose.tools import assert_equal


def test_pickler():

    hlist = list()
    for i in range(10):
        hlist.append(Hist(10, 0, 10))

    with TemporaryFile() as tmpfile:
        dump(hlist, tmpfile)
        hlist_out = load(tmpfile)
        assert_equal([h.name for h in hlist_out], [h.name for h in hlist])

    hdict = dict()
    for i in range(100):
        hist = Hist(10, 0, 1, type=random.choice('CSIFD'))
        hdict[hist.name] = hist

    with TemporaryFile() as tmpfile:
        rdir = tmpfile.mkdir('pickle')
        dump(hdict, rdir)
        hdict_out = load(rdir)
        assert_equal(len(hdict_out), 100)
        for name, hist in hdict_out.items():
            assert_equal(name, hist.name)
            assert_equal(hist.TYPE, hdict[hist.name].TYPE)


if __name__ == "__main__":
    import nose
    nose.runmodule()
