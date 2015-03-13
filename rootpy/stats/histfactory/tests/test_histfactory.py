# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from nose.plugins.skip import SkipTest

try:
    from rootpy.stats import mute_roostats; mute_roostats()
except ImportError:
    raise SkipTest("ROOT is not compiled with RooFit and RooStats enabled")

from rootpy.io import TemporaryFile
from rootpy.plotting import Hist
from rootpy.decorators import requires_ROOT
from rootpy.stats.histfactory import *
from rootpy.stats import histfactory

from nose.plugins.attrib import attr
from nose.tools import assert_raises, assert_equal, assert_true


def get_random_hist():
    h = Hist(10, -5, 5)
    h.FillRandom('gaus')
    return h

@requires_ROOT(histfactory.MIN_ROOT_VERSION, exception=SkipTest)
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
    c = sum([a, b])

    # create Channels
    channel_a = Channel('VBF')
    channel_a.data = data
    channel_a.AddSample(a)

    channel_b = Channel('VBF')
    channel_b.data = data
    channel_b.AddSample(b)

    combined_channel = channel_a + channel_b
    combined_channel = sum([channel_a, channel_b])

    # create a Measurement
    meas = Measurement('MyAnalysis')
    meas.AddChannel(channel_a)

    # create the workspace containing the model
    workspace = make_workspace(meas, silence=True)
    with TemporaryFile():
        workspace.Write()

    assert_true(channel_a.GetSample(a.name) is not None)
    channel_a.RemoveSample(a.name)
    assert_true(channel_a.GetSample(a.name) is None)

    assert_true(meas.GetChannel(channel_a.name) is not None)
    meas.RemoveChannel(channel_a.name)
    assert_true(meas.GetChannel(channel_a.name) is None)

    # test split_norm_shape
    nominal = Hist(1, 0, 1)
    nominal.FillRandom('gaus')
    hsys = HistoSys('shape', high=nominal * 1.5, low=nominal * 0.9)
    norm, shape = split_norm_shape(hsys, nominal)
    assert_equal(norm.low, 0.9)
    assert_equal(norm.high, 1.5)
    assert_equal(shape.high[1].value, nominal[1].value)
    assert_equal(shape.low[1].value, nominal[1].value)


if __name__ == "__main__":
    import nose
    nose.runmodule()
