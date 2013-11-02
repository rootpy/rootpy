# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from . import MIN_ROOT_VERSION
from ...memory.keepalive import keepalive
from ...base import NamedObject
from ... import asrootpy, QROOT, ROOT_VERSION

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


class _Named(object):

    @property
    def name(self):
        return self.GetName()

    @name.setter
    def name(self, n):
        self.SetName(n)

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        return "{0}('{1}')".format(
            self.__class__.__name__, self.GetName())


class _HistNamePathFile(object):

    @property
    def hist_name(self):
        return self.GetHistoName()

    @hist_name.setter
    def hist_name(self, name):
        self.SetHistoName(name)

    @property
    def hist_path(self):
        return self.GetHistoPath()

    @hist_path.setter
    def hist_path(self, path):
        self.SetHistoPath(path)

    @property
    def hist_file(self):
        return self.GetInputFile()

    @hist_file.setter
    def hist_file(self, infile):
        self.SetInputFile(infile)


class _SampleBase(_Named, _HistNamePathFile):

    def SetHisto(self, hist):
        super(_SampleBase, self).SetHisto(hist)
        self.SetHistoName(hist.name)
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

    _ROOT = QROOT.RooStats.HistFactory.Data

    def __init__(self, name):
        # require a name
        super(Data, self).__init__()
        self.name = name


class Sample(_SampleBase, QROOT.RooStats.HistFactory.Sample):

    _ROOT = QROOT.RooStats.HistFactory.Sample

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

    def sys_names(self):
        """
        Return a list of unique systematic names from OverallSys and HistoSys
        """
        names = {}
        for osys in self.overall_sys:
            names[osys.name] = None
        for hsys in self.histo_sys:
            names[hsys.name] = None
        return names.keys()

    def iter_sys(self):
        """
        Iterate over sys_name, overall_sys, histo_sys.
        overall_sys or histo_sys may be None for any given sys_name.
        """
        names = self.sys_names()
        for name in names:
            osys = self.GetOverallSys(name)
            hsys = self.GetHistoSys(name)
            yield name, osys, hsys

    ###########################
    # HistoSys
    ###########################
    def AddHistoSys(self, *args):
        super(Sample, self).AddHistoSys(*args)
        if len(args) == 1:
            # args is a HistoSys
            keepalive(self, args[0])

    def RemoveHistoSys(self, name):
        histosys_vect = super(Sample, self).GetHistoSysList()
        ivect = histosys_vect.begin()
        for histosys in histosys_vect:
            if histosys.GetName() == name:
                histosys_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetHistoSys(self, name):
        histosys_vect = super(Sample, self).GetHistoSysList()
        for histosys in histosys_vect:
            if histosys.GetName() == name:
                return asrootpy(histosys)
        return None

    def GetHistoSysList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetHistoSysList()]

    @property
    def histo_sys(self):
        return self.GetHistoSysList()

    ###########################
    # HistoFactor
    ###########################
    def AddHistoFactor(self, *args):
        super(Sample, self).AddHistoFactor(*args)
        if len(args) == 1:
            # args is a HistoFactor
            keepalive(self, args[0])

    def RemoveHistoFactor(self, name):
        histofactor_vect = super(Sample, self).GetHistoFactorList()
        ivect = histosys_factor.begin()
        for histofactor in histofactor_vect:
            if histofactor.GetName() == name:
                histofactor_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetHistoFactor(self, name):
        histofactor_vect = super(Sample, self).GetHistoFactorList()
        for histofactor in histofactor_vect:
            if histofactor.GetName() == name:
                return asrootpy(histofactor)
        return None

    def GetHistoFactorList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetHistoFactorList()]

    @property
    def histo_factors(self):
        return self.GetHistoFactorList()

    ###########################
    # NormFactor
    ###########################
    def AddNormFactor(self, *args):
        super(Sample, self).AddNormFactor(*args)
        if len(args) == 1:
            # args is a NormFactor
            keepalive(self, args[0])

    def RemoveNormFactor(self, name):
        normfactor_vect = super(Sample, self).GetNormFactorList()
        ivect = normfactor_vect.begin()
        for normfactor in normfactor_vect:
            if normfactor.GetName() == name:
                normfactor_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetNormFactor(self, name):
        normfactor_vect = super(Sample, self).GetNormFactorList()
        for normfactor in normfactor_vect:
            if normfactor.GetName() == name:
                return asrootpy(normfactor)
        return None

    def GetNormFactorList(self):
        return [asrootpy(norm) for norm in
                super(Sample, self).GetNormFactorList()]

    @property
    def norm_factors(self):
        return self.GetNormFactorList()

    ###########################
    # OverallSys
    ###########################
    def AddOverallSys(self, *args):
        super(Sample, self).AddOverallSys(*args)
        if len(args) == 1:
            # args is a OverallSys
            keepalive(self, args[0])

    def RemoveOverallSys(self, name):
        overallsys_vect = super(Sample, self).GetOverallSysList()
        ivect = overallsys_vect.begin()
        for overallsys in overallsys_vect:
            if overallsys.GetName() == name:
                overallsys_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetOverallSys(self, name):
        overallsys_vect = super(Sample, self).GetOverallSysList()
        for overallsys in overallsys_vect:
            if overallsys.GetName() == name:
                return asrootpy(overallsys)
        return None

    def GetOverallSysList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetOverallSysList()]

    @property
    def overall_sys(self):
        return self.GetOverallSysList()

    ###########################
    # ShapeFactor
    ###########################
    def AddShapeFactor(self, shapefactor):
        super(Sample, self).AddShapeFactor(shapefactor)
        if isinstance(shapefactor, ROOT.RooStats.HistFactory.ShapeFactor):
            keepalive(self, shapefactor)

    def RemoveShapeFactor(self, name):
        shapefactor_vect = super(Sample, self).GetShapeFactorList()
        ivect = shapefactor_vect.begin()
        for shapefactor in shapefactor_vect:
            if shapefactor.GetName() == name:
                shapefactor_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetShapeFactor(self, name):
        shapefactor_vect = super(Sample, self).GetShapeFactorList()
        for shapefactor in shapefactor_vect:
            if shapefactor.GetName() == name:
                return asrootpy(shapefactor)
        return None

    def GetShapeFactorList(self):
        return [asrootpy(sf) for sf in
                super(Sample, self).GetShapeFactorList()]

    @property
    def shape_factors(self):
        return self.GetShapeFactorList()

    ###########################
    # ShapeSys
    ###########################
    def AddShapeSys(self, *args):
        super(Sample, self).AddShapeSys(*args)
        if len(args) == 1:
            # args is a ShapeSys
            keepalive(self, args[0])

    def RemoveShapeSys(self, name):
        shapesys_vect = super(Sample, self).GetShapeSysList()
        ivect = shapesys_vect.begin()
        for shapesys in shapesys_vect:
            if shapesys.GetName() == name:
                shapesys_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetShapeSys(self, name):
        shapesys_vect = super(Sample, self).GetShapeSysList()
        for shapesys in shapesys_vect:
            if shapesys.GetName() == name:
                return asrootpy(shapesys)
        return None

    def GetShapeSysList(self):
        return [asrootpy(ss) for ss in
                super(Sample, self).GetShapeSysList()]

    @property
    def shape_sys(self):
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
    def low_name(self):
        return self.GetHistoNameLow()

    @low_name.setter
    def low_name(self, name):
        self.SetHistoNameLow(name)

    @property
    def high_name(self):
        return self.GetHistoNameHigh()

    @high_name.setter
    def high_name(self, name):
        self.SetHistoNameHigh(name)

    @property
    def low_path(self):
        return self.GetHistoPathLow()

    @low_path.setter
    def low_path(self, path):
        self.SetHistoPathLow(path)

    @property
    def high_path(self):
        return self.GetHistoPathHigh()

    @high_path.setter
    def high_path(self, path):
        self.SetHistoPathHigh(path)

    @property
    def low_file(self):
        return self.GetInputFileLow()

    @low_file.setter
    def low_file(self, infile):
        self.SetInputFileLow(infile)

    @property
    def high_file(self):
        return self.GetInputFileHigh()

    @high_file.setter
    def high_file(self, infile):
        self.SetInputFileHigh(infile)


class HistoSys(_Named, _HistoSysBase, QROOT.RooStats.HistFactory.HistoSys):

    _ROOT = QROOT.RooStats.HistFactory.HistoSys

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

    _ROOT = QROOT.RooStats.HistFactory.HistoFactor

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

    _ROOT = QROOT.RooStats.HistFactory.NormFactor

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

    _ROOT = QROOT.RooStats.HistFactory.OverallSys

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

    _ROOT = QROOT.RooStats.HistFactory.ShapeFactor

    def __init__(self, name):
        # require a name
        super(ShapeFactor, self).__init__()
        self.name = name


class ShapeSys(_Named, _HistNamePathFile, QROOT.RooStats.HistFactory.ShapeSys):

    _ROOT = QROOT.RooStats.HistFactory.ShapeSys

    def __init__(self, name):
        # require a name
        super(ShapeSys, self).__init__()
        self.name = name

    def GetErrorHist(self):
        return asrootpy(super(ShapeSys, self).GetErrorHist())

    def SetErrorHist(self, hist):
        super(ShapeSys, self).SetErrorHist(hist)
        self.SetHistoName(hist.name)
        keepalive(self, hist)

    @property
    def hist(self):
        self.GetErrorHist()

    @hist.setter
    def hist(self, h):
        self.SetErrorHist(h)


class Channel(_Named, QROOT.RooStats.HistFactory.Channel):

    _ROOT = QROOT.RooStats.HistFactory.Channel

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

    def GetSample(self, name):
        samples = super(Channel, self).GetSamples()
        for sample in samples:
            if sample.GetName() == name:
                return asrootpy(sample)
        return None

    def GetSamples(self):
        return [asrootpy(s) for s in super(Channel, self).GetSamples()]

    def AddAdditionalData(self, data):
        super(Channel, self).AddAdditionalData(data)
        keepalive(self, data)

    def GetAdditionalData(self):
        return [asrootpy(d) for d in super(Channel, self).GetAdditionalData()]

    @property
    def samples(self):
        return self.GetSamples()

    @property
    def additional_data(self):
        return self.GetAdditionalData()

    @property
    def hist_path(self):
        return self.GetHistoPath()

    @hist_path.setter
    def hist_path(self, path):
        self.SetHistoPath(path)

    @property
    def hist_file(self):
        return self.GetInputFile()

    @hist_file.setter
    def hist_file(self, infile):
        self.SetInputFile(infile)


class Measurement(NamedObject, QROOT.RooStats.HistFactory.Measurement):

    _ROOT = QROOT.RooStats.HistFactory.Measurement

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

    @property
    def channels(self):
        return self.GetChannels()
