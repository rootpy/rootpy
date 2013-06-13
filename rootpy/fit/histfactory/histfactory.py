# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from . import log; log = log[__name__]
from ...memory.keepalive import keepalive
from ... import asrootpy, QROOT, stl
import ROOT

__all__ = [
    'Data',
    'Sample',
    'HistoSys',
    'NormFactor',
    'Channel',
]

# generate required dictionaries
stl.vector('RooStats::HistFactory::HistoSys',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::NormFactor',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')
stl.vector('RooStats::HistFactory::Sample')
stl.vector('RooStats::HistFactory::Data')


class _Named(object):

    @property
    def name(self):
        return self.GetName()

    @name.setter
    def name(self, n):
        self.SetName(n)


class _SampleBase(_Named):

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
            raise ValueError('attempting to add samples with different names')
        hist1 = self.GetHisto()
        hist2 = other.GetHisto()
        hist3 = hist1 + hist2
        hist3.name = '%s_plus_%s' % (hist1.name, hist2.name)
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
        sample = super(Sample, self).__add__(other)
        # sum the histosys
        syslist1 = self.GetHistoSysList()
        syslist2 = other.GetHistoSysList()
        if len(syslist1) != len(syslist2):
            raise ValueError(
                'attempting to sum Samples with HistoSys lists of '
                'differing lengths')
        for sys1, sys2 in zip(syslist1, syslist2):
            sample.AddHistoSys(sys1 + sys2)
        # include the normfactors
        norms1 = self.GetNormFactorList()
        norms2 = other.GetNormFactorList()
        if len(norms1) != len(norms2):
            raise ValueError(
                'attempting to sum Samples with NormFactor lists of '
                'differing lengths')
        for norm1, norm2 in zip(norms1, norms2):
            if norm1.name != norm2.name:
                raise ValueError(
                    'attempting to sum Samples containing NormFactors '
                    'with differing names: {0}, {1}'.format(
                        norm1.name, norm2.name))
            # TODO check value, low and high
            sample.AddNormFactor(norm1)
        return sample

    def AddHistoSys(self, histosys):
        super(Sample, self).AddHistoSys(histosys)
        keepalive(self, histosys)

    def GetHistoSysList(self):
        return [asrootpy(syst) for syst in
                super(Sample, self).GetHistoSysList()]

    @property
    def histosys(self):
        return self.GetHistoSysList()

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


class HistoSys(_Named, QROOT.RooStats.HistFactory.HistoSys):

    def __init__(self, name, low=None, high=None):
        # require a name
        super(HistoSys, self).__init__(name)
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

    def SetHistoHigh(self, hist):
        super(HistoSys, self).SetHistoHigh(hist)
        self.SetHistoNameHigh(hist.name)
        keepalive(self, hist)

    def SetHistoLow(self, hist):
        super(HistoSys, self).SetHistoLow(hist)
        self.SetHistoNameLow(hist.name)
        keepalive(self, hist)

    def GetHistoHigh(self):
        return asrootpy(super(HistoSys, self).GetHistoHigh())

    def GetHistoLow(self):
        return asrootpy(super(HistoSys, self).GetHistoLow())

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

    def __add__(self, other):

        if self.name != other.name:
            raise ValueError('attempting to add HistoSys with different names')
        histosys = HistoSys(self.name)
        low = self.low + other.low
        low.name = '%s_plus_%s' % (self.low.name, other.low.name)
        histosys.low = low
        high = self.high + other.high
        high.name = '%s_plus_%s' % (self.high.name, other.high.name)
        histosys.high = high
        return histosys


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


class Channel(_Named, QROOT.RooStats.HistFactory.Channel):

    def __init__(self, name, inputfile=""):
        # require a name
        super(Channel, self).__init__(name, inputfile)

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
