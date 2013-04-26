# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from ROOT import TH1D
from rootpy.plotting import Hist, Legend


def test_init():

    l = Legend(2)
    h = Hist(10, 0, 1)
    l.AddEntry(h)
    hr = TH1D("test", "", 10, 0, 1)
    l.AddEntry(hr)


if __name__ == "__main__":
    import nose
    nose.runmodule()
