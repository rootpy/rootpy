from . import log; log = log[__name__]
from ...memory.keepalive import keepalive
from ... import asrootpy
from . import Channel, Measurement, HistoSys, OverallSys
import ROOT

__all__ = [
    'make_channel',
    'make_measurement',
    'make_models',
    'make_model',
    'make_workspace',
    'split_norm_shape',
]


def make_channel(name, samples, data=None):
    """
    Create a Channel from a list of Samples
    """
    llog = log['make_channel']
    llog.info("creating channel %s" % name)
    # avoid segfault if name begins with a digit by using "channel_" prefix
    chan = Channel('channel_%s' % name)
    chan.SetStatErrorConfig(0.05, "Poisson")

    if data is not None:
        llog.info("setting data")
        chan.SetData(data)

    for sample in samples:
        llog.info("adding sample {0}".format(sample.GetName()))
        chan.AddSample(sample)

    return chan


def make_measurement(name,
                     channels,
                     lumi=1.0, lumi_rel_error=0.,
                     output_prefix='./histfactory',
                     POI=None,
                     const_params=None):
    """
    Create a Measurement from a list of Channels
    """
    llog = log['make_measurement']
    # Create the measurement
    llog.info("creating measurement {0}".format(name))
    meas = Measurement('measurement_{0}'.format(name), '')

    meas.SetOutputFilePrefix(output_prefix)
    if POI is not None:
        if isinstance(POI, basestring):
            llog.info("setting POI {0}".format(POI))
            meas.SetPOI(POI)
        else:
            llog.info("adding POIs {0}".format(', '.join(POI)))
            for p in POI:
                meas.AddPOI(p)

    llog.info("setting lumi={0:f} +/- {1:f}".format(lumi, lumi_rel_error))
    meas.lumi = lumi
    meas.lumi_rel_error = lumi_rel_error

    for channel in channels:
        llog.info("adding channel {0}".format(channel.GetName()))
        meas.AddChannel(channel)

    if const_params is not None:
        llog.info("adding constant parameters {0}".format(
            ', '.join(const_params)))
        for param in const_params:
            meas.AddConstantParam(param)

    return meas


def make_models(measurement):
    """
    Create a workspace containing all models for a Measurement
    """
    return asrootpy(ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(measurement))


def make_model(measurement, channel=None):
    """
    Create a workspace containing the model for a measurement

    If `channel` is None then include all channels in the model
    """
    hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(measurement)
    if channel is not None:
        workspace = hist2workspace.MakeSingleChannelModel(measurement, channel)
    else:
        workspace = hist2workspace.MakeCombinedModel(measurement)
    workspace = asrootpy(workspace)
    keepalive(workspace, measurement)
    return workspace


def make_workspace(name, channels, **kwargs):
    """
    Create a workspace from a list of channels

    kwargs are passed to `make_measurement`
    """
    if not isinstance(channels, (list, tuple)):
        channels = [channels]
    measurement = make_measurement(name, channels, **kwargs)
    workspace = make_model(measurement)
    workspace.SetName('workspace_%s' % name)
    return workspace, measurement


def split_norm_shape(histosys, nominal_hist):
    """
    Split a HistoSys into normalization (OverallSys) and shape (HistoSys)
    components.

    It is recommended to use OverallSys as much as possible, which tries to
    enforce continuity up to the second derivative during
    interpolation/extrapolation. So, if there is indeed a shape variation, then
    factorize it into shape and normalization components.
    """
    up = histosys.GetHistoHigh()
    dn = histosys.GetHistoLow()
    up = up.Clone(name=up.name + '_shape')
    dn = dn.Clone(name=dn.name + '_shape')
    n_nominal = nominal_hist.Integral(0, nominal_hist.GetNbinsX() + 1)
    n_up = up.Integral(0, up.GetNbinsX() + 1)
    n_dn = dn.Integral(0, dn.GetNbinsX() + 1)
    up.Scale(n_nominal / n_up)
    dn.Scale(n_nominal / n_dn)
    shape = HistoSys(histosys.GetName(), low=dn, high=up)
    norm = OverallSys(histosys.GetName(),
                      low=n_dn / n_nominal,
                      high=n_up / n_nominal)
    return norm, shape
