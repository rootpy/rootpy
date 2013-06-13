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
]

# generate required dictionaries
stl.vector('RooStats::HistFactory::HistoSys',
           headers='<vector>;<RooStats/HistFactory/Systematics.h>')


class _SampleBase(object):

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
        hist1 = self.GetHisto()
        hist2 = other.GetHisto()
        hist3 = hist1 + hist2
        hist3.name = '%s_plus_%s' % (hist1.name, hist2.name)
        sample = self.__class__()
        sample.SetHisto(hist3)
        return sample


class Data(_SampleBase, QROOT.RooStats.HistFactory.Data):
    pass


class Sample(_SampleBase, QROOT.RooStats.HistFactory.Sample):

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


class HistoSys(QROOT.RooStats.HistFactory.HistoSys):

    def __init__(self, name):
        # require a name
        super(HistoSys, self).__init__(name)

    def SetHistoHigh(self, hist):
        super(HistoSys, self).SetHistoHigh(hist)
        keepalive(self, hist)

    def SetHistoLow(self, hist):
        super(HistoSys, self).SetHistoLow(hist)
        keepalive(self, hist)

    def GetHistoHigh(self):
        return asrootpy(super(HistoSys, self).GetHistoHigh())

    def GetHistoLow(self):
        return asrootpy(super(HistoSys, self).GetHistoLow())

    @property
    def name(self):
        return self.GetName()

    @name.setter
    def name(self, n):
        self.SetName(n)

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
