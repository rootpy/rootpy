# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.io import root_open, TemporaryFile
from rootpy.io.pickler import load, dump
from rootpy.plotting import Hist
import random
import tempfile
from nose.tools import assert_equal, assert_true, assert_false


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


def test_pickler_proxy():
    h = Hist(5, 0, 1, name='hist')
    f = tempfile.NamedTemporaryFile(suffix='.root')

    with root_open(f.name, 'recreate') as outfile:
        dump([h], outfile)

    class IsCalled(object):
        def __init__(self, func):
            self.func = func
            self.called = False

        def __call__(self, path):
            if path != '_pickle;1':
                self.called = True
            return self.func(path)

    with root_open(f.name) as infile:
        infile.Get = IsCalled(infile.Get)
        hlist = load(infile, use_proxy=False)
        assert_true(infile.Get.called)

    with root_open(f.name) as infile:
        infile.Get = IsCalled(infile.Get)
        hlist = load(infile, use_proxy=True)
        assert_false(infile.Get.called)
        assert_equal(hlist[0].name, 'hist')
        assert_true(infile.Get.called)

    f.close()


if __name__ == "__main__":
    import nose
    nose.runmodule()
