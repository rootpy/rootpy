from . import log; log = log[__name__]
from ...memory.keepalive import keepalive
from ... import asrootpy
from . import Channel, Measurement
import ROOT

__all__ = [
    'make_channel',
    'make_measurement',
    'make_models',
    'make_model',
    'make_workspace',
]


def make_channel(name, samples, data=None):
    """
    Create a Channel from a list of Samples
    """
    log.info("creating channel %s" % name)
    # avoid segfault if name begins with a digit by using "channel_" prefix
    chan = Channel('channel_%s' % name)
    chan.SetStatErrorConfig(0.05, "Poisson")

    if data is not None:
        log.info("setting data")
        chan.SetData(data)

    for sample in samples:
        log.info("adding sample %s" % sample.GetName())
        chan.AddSample(sample)

    return chan


def make_measurement(name,
                     channels,
                     lumi=1.0, lumi_rel_error=0.,
                     output_prefix='./histfactory',
                     POI=None):
    """
    Create a Measurement from a list of Channels
    """
    # Create the measurement
    log.info("creating measurement %s" % name)
    meas = Measurement('measurement_%s' % name, '')

    meas.SetOutputFilePrefix(output_prefix)
    if POI is not None:
        if isinstance(POI, basestring):
            log.info("setting POI %s" % POI)
            meas.SetPOI(POI)
        else:
            for p in POI:
                log.info("adding POI %s" % p)
                meas.AddPOI(p)

    log.info("setting lumi=%f +/- %f" % (lumi, lumi_rel_error))
    meas.lumi = lumi
    meas.lumi_rel_error = lumi_rel_error
    # TODO: is this correct?
    #meas.AddConstantParam('Lumi')

    for channel in channels:
        log.info("adding channel %s" % channel.GetName())
        meas.AddChannel(channel)

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
