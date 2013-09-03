# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from . import log; log = log[__name__]
from . import MIN_ROOT_VERSION
from ...memory.keepalive import keepalive
from ...core import NamedObject
from ... import asrootpy, QROOT, stl, ROOT_VERSION
import ROOT

if ROOT_VERSION < MIN_ROOT_VERSION:
    raise NotImplementedError(
        "histfactory requires ROOT {0} but you are using {1}".format(
            MIN_ROOT_VERSION, ROOT_VERSION))

__all__ = [
    'Data',
    'Sample',
    'HistoSys',
    'HistoFactor',
    'NormFactor',
    'OverallSys',
    'ShapeFactor',
    'ShapeSys',
    'Channel',
    'Measurement',
]

# generate required dictionaries
stl.vector('RooStats::HistFactory::HistoSys',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::HistoFactor',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::NormFactor',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::OverallSys',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::ShapeFactor',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::ShapeSys',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::Sample')
stl.vector('RooStats::HistFactory::Data')
stl.vector('RooStats::HistFactory::Channel')
stl.vector('RooStats::HistFactory::Measurement')


class _Named(object):

    @property
    def name(self):
        return self.GetName()

    @name.setter
    def name(self, n):
        self.SetName(n)


class _NamePathFile(object):

    @property
    def histname(self):
        return self.GetHistoName()

    @histname.setter
    def histname(self, name):
        self.SetHistoName(name)

    @property
    def path(self):
        return self.GetHistoPath()

    @path.setter
    def path(self, path):
        self.SetHistoPath(path)

    @property
    def file(self):
        return self.GetInputFile()

    @file.setter
    def file(self, infile):
        self.SetInputFile(infile)


class _SampleBase(_Named, _NamePathFile):

    def SetHisto(self, hist):
        super(_SampleBase, self).SetHisto(hist)
        keepalive(self, hist)

    def GetHisto(self):
        return asrootpy(super(_SampleBase, self).GetHisto())

    @property
    def hist(self):
        return self.GetHisto()

    @hist.setter
    def hist(self, h):
        self.SetHisto(h)

    def __add__(self, other):
        if self.name != other.name:
            raise ValueError("attempting to add samples with different names")
        hist1 = self.GetHisto()
        hist2 = other.GetHisto()
        hist3 = hist1 + hist2
        hist3.name = '{0}_plus_{1}'.format(hist1.name, hist2.name)
        sample = self.__class__(self.name)
        sample.SetHisto(hist3)
        return sample


class Data(_SampleBase, QROOT.RooStats.HistFactory.Data):

    def __init__(self, name):
        # require a name
        super(Data, self).__init__()
        self.name = name


class Sample(_SampleBase, QROOT.RooStats.HistFactory.Sample):

    def __init__(self, name):
        # require a sample name
        super(Sample, self).__init__(name)

    def __add__(self, other):

        if self.GetHistoFactorList() or other.GetHistoFactorList():
            raise NotImplementedError(
                "Samples cannot be summed if "
                "they contain HistoFactors")

        if self.GetShapeFactorList() or other.GetShapeFactorList():
            raise NotImplementedError(
                "Samples cannot be summed if "
                "they contain ShapeFactors")

        if self.GetShapeSysList() or other.GetShapeSysList():
            raise NotImplementedError(
                "Samples cannot be summed if "
                "they contain ShapeSys")

        if self.GetNormalizeByTheory() != other.GetNormalizeByTheory():
            raise ValueError(
                "attempting to sum samples with "
                "inconsistent NormalizeByTheory")

        sample = super(Sample, self).__add__(other)
        sample.SetNormalizeByTheory(self.GetNormalizeByTheory())

        # sum the histosys
        syslist1 = self.GetHistoSysList()
        syslist2 = other.GetHistoSysList()
        if len(syslist1) != len(syslist2):
            raise ValueError(
                "attempting to sum Samples with HistoSys lists of "
                "differing lengths")
        for sys1, sys2 in zip(syslist1, syslist2):
            sample.AddHistoSys(sys1 + sys2)

        # include the overallsys
        overall1 = self.GetOverallSysList()
        overall2 = other.GetOverallSysList()
        if len(overall1) != len(overall2):
            raise ValueError(
                "attempting to sum Samples with OverallSys lists of "
                "differing lengths")
        for o1, o2 in zip(overall1, overall2):
            if o1.name != o2.name:
                raise ValueError(
                    "attempting to sum Samples containing OverallSys "
                    "with differing names: {0}, {1}".format(
                        o1.name, o2.name))
            # TODO check equality of value, low and high
            sample.AddOverallSys(o1)

        # include the normfactors
        norms1 = self.GetNormFactorList()
        norms2 = other.GetNormFactorList()
        if len(norms1) != len(norms2):
            raise ValueError(
                "attempting to sum Samples with NormFactor lists of "
                "differing lengths")
        for norm1, norm2 in zip(norms1, norms2):
            if norm1.name != norm2.name:
                raise ValueError(
                    "attempting to sum Samples containing NormFactors "
                    "with differing names: {0}, {1}".format(
                        norm1.name, norm2.name))
            # TODO check equality of value, low and high
            sample.AddNormFactor(norm1)
        return sample

    def __radd__(self, other):
        # support sum([list of Samples])
        if other == 0:
            return self
        raise TypeError(
            "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                other.__class__.__name__, self.__class__.__name__))

    ###########################
    # HistoSys
    ###########################
    def AddHistoSys(self, *args):
        super(Sample, self).AddHistoSys(*args)
        if len(args) == 1:
            # args is a HistoSys
            keepalive(self, args[0])

    def GetHistoSysList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetHistoSysList()]

    @property
    def histosys(self):
        return self.GetHistoSysList()

    ###########################
    # HistoFactor
    ###########################
    def AddHistoFactor(self, *args):
        super(Sample, self).AddHistoFactor(*args)
        if len(args) == 1:
            # args is a HistoFactor
            keepalive(self, args[0])

    def GetHistoFactorList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetHistoFactorList()]

    @property
    def histofactors(self):
        return self.GetHistoFactorList()

    ###########################
    # NormFactor
    ###########################
    def AddNormFactor(self, *args):
        super(Sample, self).AddNormFactor(*args)
        if len(args) == 1:
            # args is a NormFactor
            keepalive(self, args[0])

    def GetNormFactorList(self):
        return [asrootpy(norm) for norm in
                super(Sample, self).GetNormFactorList()]

    @property
    def normfactors(self):
        return self.GetNormFactorList()

    ###########################
    # OverallSys
    ###########################
    def AddOverallSys(self, *args):
        super(Sample, self).AddOverallSys(*args)
        if len(args) == 1:
            # args is a OverallSys
            keepalive(self, args[0])

    def GetOverallSysList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetOverallSysList()]

    @property
    def overallsys(self):
        return self.GetOverallSysList()

    ###########################
    # ShapeFactor
    ###########################
    def AddShapeFactor(self, shapefactor):
        super(Sample, self).AddShapeFactor(shapefactor)
        if isinstance(shapefactor, ROOT.RooStats.HistFactory.ShapeFactor):
            keepalive(self, shapefactor)

    def GetShapeFactorList(self):
        return [asrootpy(sf) for sf in
                super(Sample, self).GetShapeFactorList()]

    @property
    def shapefactors(self):
        return self.GetShapeFactorList()

    ###########################
    # ShapeSys
    ###########################
    def AddShapeSys(self, *args):
        super(Sample, self).AddShapeSys(*args)
        if len(args) == 1:
            # args is a ShapeSys
            keepalive(self, args[0])

    def GetShapeSysList(self):
        return [asrootpy(ss) for ss in
                super(Sample, self).GetShapeSysList()]

    @property
    def shapesys(self):
        return self.GetShapeSysList()


class _HistoSysBase(object):

    def SetHistoHigh(self, hist):
        super(_HistoSysBase, self).SetHistoHigh(hist)
        self.SetHistoNameHigh(hist.name)
        keepalive(self, hist)

    def SetHistoLow(self, hist):
        super(_HistoSysBase, self).SetHistoLow(hist)
        self.SetHistoNameLow(hist.name)
        keepalive(self, hist)

    def GetHistoHigh(self):
        return asrootpy(super(_HistoSysBase, self).GetHistoHigh())

    def GetHistoLow(self):
        return asrootpy(super(_HistoSysBase, self).GetHistoLow())

    @property
    def low(self):
        return self.GetHistoLow()

    @low.setter
    def low(self, h):
        self.SetHistoLow(h)

    @property
    def high(self):
        return self.GetHistoHigh()

    @high.setter
    def high(self, h):
        self.SetHistoHigh(h)

    @property
    def lowname(self):
        return self.GetHistoNameLow()

    @lowname.setter
    def lowname(self, name):
        self.SetHistoNameLow(name)

    @property
    def highname(self):
        return self.GetHistoNameHigh()

    @highname.setter
    def highname(self, name):
        self.SetHistoNameHigh(name)

    @property
    def lowpath(self):
        return self.GetHistoPathLow()

    @lowpath.setter
    def lowpath(self, path):
        self.SetHistoPathLow(path)

    @property
    def highpath(self):
        return self.GetHistoPathHigh()

    @highpath.setter
    def highpath(self, path):
        self.SetHistoPathHigh(path)

    @property
    def lowfile(self):
        return self.GetInputFileLow()

    @lowfile.setter
    def lowfile(self, infile):
        self.SetInputFileLow(infile)

    @property
    def highfile(self):
        return self.GetInputFileHigh()

    @highfile.setter
    def highfile(self, infile):
        self.SetInputFileHigh(infile)


class HistoSys(_Named, _HistoSysBase, QROOT.RooStats.HistFactory.HistoSys):

    def __init__(self, name, low=None, high=None):
        # require a name
        super(HistoSys, self).__init__(name)
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

    def __add__(self, other):

        if self.name != other.name:
            raise ValueError("attempting to add HistoSys with different names")
        histosys = HistoSys(self.name)
        low = self.low + other.low
        low.name = '{0}_plus_{1}'.format(self.low.name, other.low.name)
        histosys.low = low
        high = self.high + other.high
        high.name = '{0}_plus_{1}'.format(self.high.name, other.high.name)
        histosys.high = high
        return histosys


class HistoFactor(_Named, _HistoSysBase,
                  QROOT.RooStats.HistFactory.HistoFactor):

    def __init__(self, name, low=None, high=None):
        # require a name
        super(HistoFactor, self).__init__(name)
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

    def __add__(self, other):

        raise NotImplementedError("HistoFactors cannot be summed")


class NormFactor(_Named, QROOT.RooStats.HistFactory.NormFactor):

    def __init__(self, name, value=None, low=None, high=None, const=None):

        super(NormFactor, self).__init__()
        self.name = name
        if value is not None:
            self.value = value
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        if const is not None:
            self.const = const

    @property
    def const(self):
        return self.GetConst()

    @const.setter
    def const(self, value):
        self.SetConst(value)

    @property
    def value(self):
        return self.GetVal()

    @value.setter
    def value(self, value):
        self.SetVal(value)

    @property
    def low(self):
        return self.GetLow()

    @low.setter
    def low(self, value):
        self.SetLow(value)

    @property
    def high(self):
        return self.GetHigh()

    @high.setter
    def high(self, value):
        self.SetHigh(value)


class OverallSys(_Named, QROOT.RooStats.HistFactory.OverallSys):

    def __init__(self, name, low=None, high=None):
        # require a name
        super(OverallSys, self).__init__()
        self.name = name
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

    @property
    def low(self):
        return self.GetLow()

    @low.setter
    def low(self, value):
        self.SetLow(value)

    @property
    def high(self):
        return self.GetHigh()

    @high.setter
    def high(self, value):
        self.SetHigh(value)


class ShapeFactor(_Named, QROOT.RooStats.HistFactory.ShapeFactor):

    def __init__(self, name):
        # require a name
        super(ShapeFactor, self).__init__()
        self.name = name


class ShapeSys(_Named, _NamePathFile, QROOT.RooStats.HistFactory.ShapeSys):

    def __init__(self, name):
        # require a name
        super(ShapeSys, self).__init__()
        self.name = name

    def GetErrorHist(self):
        return asrootpy(super(ShapeSys, self).GetErrorHist())

    def SetErrorHist(self, hist):
        super(ShapeSys, self).SetErrorHist(hist)
        keepalive(self, hist)

    @property
    def hist(self):
        self.GetErrorHist()

    @hist.setter
    def hist(self, h):
        self.SetErrorHist(h)


class Channel(_Named, QROOT.RooStats.HistFactory.Channel):

    def __init__(self, name, inputfile=""):
        # require a name
        super(Channel, self).__init__(name, inputfile)

    def __add__(self, other):
        channel = Channel('{0}_plus_{1}'.format(self.name, other.name))
        channel.SetData(self.data + other.data)
        samples1 = self.samples
        samples2 = other.samples
        if len(samples1) != len(samples2):
            raise ValueError(
                "attempting to add Channels containing differing numbers of "
                "Samples")
        for s1, s2 in zip(samples1, samples2):
            # samples must be compatible
            channel.AddSample(s1 + s2)
        channel.SetStatErrorConfig(self.GetStatErrorConfig())
        return channel

    def __radd__(self, other):
        # support sum([list of Channels])
        if other == 0:
            return self
        raise TypeError(
            "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                other.__class__.__name__, self.__class__.__name__))

    def SetData(self, data):
        super(Channel, self).SetData(data)
        if isinstance(data, ROOT.TH1):
            keepalive(self, data)

    def GetData(self):
        return asrootpy(super(Channel, self).GetData())

    @property
    def data(self):
        return self.GetData()

    @data.setter
    def data(self, d):
        self.SetData(d)

    def AddSample(self, sample):
        super(Channel, self).AddSample(sample)
        keepalive(self, sample)

    def AddAdditionalData(self, data):
        super(Channel, self).AddAdditionalData(data)
        keepalive(self, data)

    def GetSamples(self):
        return [asrootpy(s) for s in super(Channel, self).GetSamples()]

    def GetAdditionalData(self):
        return [asrootpy(d) for d in super(Channel, self).GetAdditionalData()]

    @property
    def samples(self):
        return self.GetSamples()

    @property
    def additionaldata(self):
        return self.GetAdditionalData()

    @property
    def path(self):
        return self.GetHistoPath()

    @path.setter
    def path(self, path):
        self.SetHistoPath(path)

    @property
    def file(self):
        return self.GetInputFile()

    @file.setter
    def file(self, infile):
        self.SetInputFile(infile)


class Measurement(NamedObject, QROOT.RooStats.HistFactory.Measurement):

    def __init__(self, name, title=""):
        # require a name
        super(Measurement, self).__init__(name=name, title=title)
        self.SetExportOnly(True)

    @property
    def lumi(self):
        return self.GetLumi()

    @lumi.setter
    def lumi(self, l):
        self.SetLumi(l)

    @property
    def lumi_rel_error(self):
        return self.GetLumiRelErr()

    @lumi_rel_error.setter
    def lumi_rel_error(self, err):
        self.SetLumiRelErr(err)

    @property
    def poi(self):
        return list(self.GetPOIList())

    @poi.setter
    def poi(self, p):
        # this also adds a new POI so calling this multiple times will add
        # multiple POIs
        self.SetPOI(p)

    def AddChannel(self, channel):
        super(Measurement, self).AddChannel(channel)
        keepalive(self, channel)

    def GetChannel(self, name):
        return asrootpy(super(Measurement, self).GetChannel(name))

    def GetChannels(self):
        return [asrootpy(c) for c in super(Measurement, self).GetChannels()]
