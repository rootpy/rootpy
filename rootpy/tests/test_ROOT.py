# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy import ROOT
from rootpy.plotting.hist import _Hist
from nose.tools import assert_equal, assert_true


def test_ROOT():

    a = ROOT.TH1F("a", "a", 10, 0, 1)
    assert_true(isinstance(a, _Hist))

    b = ROOT.Hist(10, 0, 1, type='F')
    assert_equal(a.TYPE, b.TYPE)


if __name__ == "__main__":
    import nose
    nose.runmodule()
