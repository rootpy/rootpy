# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Tests for the file module.
"""

from rootpy.io import TemporaryFile
from rootpy.io.pickler import load, dump
from rootpy.plotting import Hist
from nose.tools import assert_equal


def test_pickler():

    hlist = []
    for i in range(10):
        hlist.append(Hist(10, 0, 10, name='h{0}'.format(i)))

    with TemporaryFile() as tmpfile:
        dump(hlist, tmpfile)
        hlist_out = load(tmpfile)
        assert_equal([h.name for h in hlist_out], [h.name for h in hlist])


if __name__ == "__main__":
    import nose
    nose.runmodule()
