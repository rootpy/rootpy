# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist
from rootpy.fit.histfactory import Sample, HistoSys

from nose.plugins.attrib import attr
from nose.tools import assert_raises, assert_equal


def get_random_hist():
    h = Hist(10, -5, 5)
    h.FillRandom('gaus')
    return h

def test_histfactory():
    a = Sample('QCD')
    b = Sample('QCD')

    for sample in (a, b):
        sample.hist = get_random_hist()
        for sysname in ('x', 'y', 'z'):
            histosys = HistoSys(sysname)
            histosys.high = get_random_hist()
            histosys.low = get_random_hist()
            sample.AddHistoSys(histosys)

    c = a + b


if __name__ == "__main__":
    import nose
    nose.runmodule()
