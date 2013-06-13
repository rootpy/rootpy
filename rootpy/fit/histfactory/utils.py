from . import log; log = log[__name__]
from ...memory.keepalive import keepalive
import ROOT

__all__ = [
    'make_channel',
    'make_measurement',
    'make_all_models',
    'make_model',
    'make_workspace',
]


def make_channel(name, samples, data=None):

    log.info("creating channel %s" % name)
    # avoid segfault if name begins with a digit by using "channel_" prefix
    chan = ROOT.RooStats.HistFactory.Channel('channel_%s' % name)
    if data is not None:
        log.info("setting data")
        chan.SetData(data)
    chan.SetStatErrorConfig(0.05, "Poisson")

    for sample in samples:
        log.info("adding sample %s" % sample.GetName())
        chan.AddSample(sample)
    keepalive(chan, *samples)

    return chan


def make_measurement(name,
                     channels,
                     lumi=1.0, lumi_rel_error=0.,
                     output_prefix='./histfactory',
                     POI=None):

    # Create the measurement
    log.info("creating measurement %s" % name)
    meas = ROOT.RooStats.HistFactory.Measurement(
        'measurement_%s' % name, '')

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
    meas.SetLumi(lumi)
    meas.SetLumiRelErr(lumi_rel_error)
    meas.SetExportOnly(True)
    # TODO: is this correct?
    #meas.AddConstantParam('Lumi')

    for channel in channels:
        log.info("adding channel %s" % channel.GetName())
        meas.AddChannel(channel)
    keepalive(meas, *channels)

    return meas


def make_all_models(measurement):

    return ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(measurement)


def make_model(measurement, channel=None):

    hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(measurement)
    if channel is not None:
        workspace = hist2workspace.MakeSingleChannelModel(measurement, channel)
    else:
        workspace = hist2workspace.MakeCombinedModel(measurement)
    keepalive(workspace, measurement)
    return workspace


def make_workspace(name, channels,
                   lumi_rel_error=0.,
                   POI='SigXsecOverSM'):

    if not isinstance(channels, (list, tuple)):
        channels = [channels]
    measurement = make_measurement(
            name,
            channels,
            lumi_rel_error=lumi_rel_error,
            POI=POI)
    workspace = make_model(measurement)
    workspace.SetName('workspace_%s' % name)
    return workspace, measurement
