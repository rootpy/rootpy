# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist
from rootpy.fit.histfactory import *

from nose.plugins.attrib import attr
from nose.tools import assert_raises, assert_equal


def get_random_hist():
    h = Hist(10, -5, 5)
    h.FillRandom('gaus')
    return h

def test_histfactory():

    # create some Samples
    data = Data('data')
    data.hist = get_random_hist()
    a = Sample('QCD')
    b = Sample('QCD')

    for sample in (a, b):
        sample.hist = get_random_hist()
        # include some histosysts
        for sysname in ('x', 'y', 'z'):
            histosys = HistoSys(sysname)
            histosys.high = get_random_hist()
            histosys.low = get_random_hist()
            sample.AddHistoSys(histosys)
        # include some normfactors
        for normname in ('x', 'y', 'z'):
            norm = NormFactor(normname)
            norm.value = 1
            norm.high = 2
            norm.low = 0
            norm.const = False
            sample.AddNormFactor(norm)

    # samples must be compatible here
    c = a + b

    # create a Channel
    channel = Channel('VBF')
    channel.data = data
    channel.AddSample(a)

    # create a Measurement
    meas = Measurement('MyAnalysis')
    meas.AddChannel(channel)


if __name__ == "__main__":
    import nose
    nose.runmodule()
