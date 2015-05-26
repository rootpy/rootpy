# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from math import sqrt

import ROOT

from . import log; log = log[__name__]
from . import MIN_ROOT_VERSION
from ...extern.six import string_types
from ...memory.keepalive import keepalive
from ...base import NamedObject
from ... import asrootpy, QROOT, ROOT_VERSION

if ROOT_VERSION < MIN_ROOT_VERSION:
    raise NotImplementedError(
        "histfactory requires ROOT {0} but you are using {1}".format(
            MIN_ROOT_VERSION, ROOT_VERSION))

HistFactory = QROOT.RooStats.HistFactory
Constraint = HistFactory.Constraint

__all__ = [
    'Constraint',
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
        hist = super(_SampleBase, self).GetHisto()
        # NULL pointer check
        if hist == None:
            return None
        return asrootpy(hist)

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
        sample = self.__class__(self.name)
        if hist1 is not None and hist2 is not None:
            hist3 = hist1 + hist2
            hist3.name = '{0}_plus_{1}'.format(hist1.name, hist2.name)
            sample.SetHisto(hist3)
        return sample


class Data(_SampleBase, HistFactory.Data):
    _ROOT = HistFactory.Data

    def __init__(self, name, hist=None):
        # require a name
        super(Data, self).__init__()
        self.name = name
        if hist is not None:
            self.SetHisto(hist)

    def total(self, xbin1=1, xbin2=-2):
        """
        Return the total yield and its associated statistical uncertainty.
        """
        return self.hist.integral(xbin1=xbin1, xbin2=xbin2, error=True)

    def Clone(self):
        clone = Data(self.name)
        hist = self.hist
        if hist is not None:
            clone.hist = hist.Clone(shallow=True)
        return clone


class Sample(_SampleBase, HistFactory.Sample):
    _ROOT = HistFactory.Sample

    def __init__(self, name, hist=None):
        # require a sample name
        super(Sample, self).__init__(name)
        if hist is not None:
            self.SetHisto(hist)

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

    def __mul__(self, scale):
        clone = self.Clone()
        clone *= scale
        return clone

    def __imul__(self, scale):
        hist = self.hist
        if hist is not None:
            hist *= scale
        for hsys in self.histo_sys:
            low = hsys.low
            high = hsys.high
            if low is not None:
                low *= scale
            if high is not None:
                high *= scale
        return self

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

    def sys_hist(self, name=None):
        """
        Return the effective low and high histogram for a given systematic.
        If this sample does not contain the named systematic then return
        the nominal histogram for both low and high variations.
        """
        if name is None:
            low = self.hist.Clone(shallow=True)
            high = self.hist.Clone(shallow=True)
            return low, high
        osys = self.GetOverallSys(name)
        hsys = self.GetHistoSys(name)
        if osys is None:
            osys_high, osys_low = 1., 1.
        else:
            osys_high, osys_low = osys.high, osys.low
        if hsys is None:
            hsys_high = self.hist.Clone(shallow=True)
            hsys_low = self.hist.Clone(shallow=True)
        else:
            hsys_high = hsys.high.Clone(shallow=True)
            hsys_low = hsys.low.Clone(shallow=True)
        return hsys_low * osys_low, hsys_high * osys_high

    def has_sys(self, name):
        return (self.GetOverallSys(name) is not None or
                self.GetHistoSys(name) is not None)

    def total(self, xbin1=1, xbin2=-2):
        """
        Return the total yield and its associated statistical and
        systematic uncertainties.
        """
        integral, stat_error = self.hist.integral(
            xbin1=xbin1, xbin2=xbin2, error=True)
        # sum systematics in quadrature
        ups = [0]
        dns = [0]
        for sys_name in self.sys_names():
            sys_low, sys_high = self.sys_hist(sys_name)
            up = sys_high.integral(xbin1=xbin1, xbin2=xbin2) - integral
            dn = sys_low.integral(xbin1=xbin1, xbin2=xbin2) - integral
            if up > 0:
                ups.append(up**2)
            else:
                dns.append(up**2)
            if dn > 0:
                ups.append(dn**2)
            else:
                dns.append(dn**2)
        syst_error = (sqrt(sum(ups)), sqrt(sum(dns)))
        return integral, stat_error, syst_error

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

    def Clone(self):
        clone = self.__class__(self.name)
        hist = self.hist
        if hist is not None:
            clone.hist = hist.Clone(shallow=True)
        # HistoSys
        for hsys in self.histo_sys:
            clone.AddHistoSys(hsys.Clone())
        # HistoFactor
        for hfact in self.histo_factors:
            clone.AddHistoFactor(hfact.Clone())
        # NormFactor
        for norm in self.norm_factors:
            clone.AddNormFactor(norm.Clone())
        # OverallSys
        for osys in self.overall_sys:
            clone.AddOverallSys(osys.Clone())
        # ShapeFactor
        for sfact in self.shape_factors:
            clone.AddShapeFactor(sfact.Clone())
        # ShapeSys
        for ssys in self.shape_sys:
            clone.AddShapeSys(ssys.Clone())
        return clone


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
        hist = super(_HistoSysBase, self).GetHistoHigh()
        # NULL pointer check
        if hist == None:
            return None
        return asrootpy(hist)

    def GetHistoLow(self):
        hist = super(_HistoSysBase, self).GetHistoLow()
        # NULL pointer check
        if hist == None:
            return None
        return asrootpy(hist)

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

    def Clone(self):
        clone = self.__class__(self.name)
        low = self.low
        high = self.high
        if low is not None:
            clone.low = low.Clone(shallow=True)
        if high is not None:
            clone.high = high.Clone(shallow=True)
        clone.low_name = self.low_name
        clone.high_name = self.high_name
        clone.low_path = self.low_path
        clone.high_path = self.high_path
        clone.low_file = self.low_file
        clone.high_file = self.high_file
        return clone


class HistoSys(_Named, _HistoSysBase, HistFactory.HistoSys):
    _ROOT = HistFactory.HistoSys

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
                  HistFactory.HistoFactor):
    _ROOT = HistFactory.HistoFactor

    def __init__(self, name, low=None, high=None):
        # require a name
        super(HistoFactor, self).__init__(name)
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high

    def __add__(self, other):
        raise NotImplementedError("HistoFactors cannot be summed")


class NormFactor(_Named, HistFactory.NormFactor):
    _ROOT = HistFactory.NormFactor

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

    def Clone(self):
        return NormFactor(self.name,
            value=self.value,
            low=self.low,
            high=self.high,
            const=self.const)


class OverallSys(_Named, HistFactory.OverallSys):
    _ROOT = HistFactory.OverallSys

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

    def Clone(self):
        return OverallSys(self.name, low=self.low, high=self.high)


class ShapeFactor(_Named, HistFactory.ShapeFactor):
    _ROOT = HistFactory.ShapeFactor

    def __init__(self, name):
        # require a name
        super(ShapeFactor, self).__init__()
        self.name = name

    def Clone(self):
        return ShapeFactor(self.name)


class ShapeSys(_Named, _HistNamePathFile, HistFactory.ShapeSys):
    _ROOT = HistFactory.ShapeSys

    def __init__(self, name):
        # require a name
        super(ShapeSys, self).__init__()
        self.name = name
        # ConstraintType not initialized correctly on C++ side
        # ROOT.RooStats.HistFactory.Constraint.Gaussian
        super(ShapeSys, self).SetConstraintType(Constraint.Gaussian)

    def SetConstraintType(self, value):
        _value = value.lower() if isinstance(value, string_types) else value
        if _value in (Constraint.Gaussian, 'gauss', 'gaussian'):
            super(ShapeSys, self).SetConstraintType(Constraint.Gaussian)
        elif _value in (Constraint.Poisson, 'pois', 'poisson'):
            super(ShapeSys, self).SetConstraintType(Constraint.Poisson)
        else:
            raise ValueError(
                "'{0}' is not a valid constraint".format(value))

    @property
    def constraint(self):
        return super(ShapeSys, self).GetConstraintType()

    @constraint.setter
    def constraint(self, value):
        self.SetConstraintType(value)

    def GetErrorHist(self):
        hist = super(ShapeSys, self).GetErrorHist()
        # NULL pointer check
        if hist == None:
            return None
        return asrootpy(hist)

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

    def Clone(self):
        clone = ShapeSys(self.name)
        hist = self.hist
        if hist is not None:
            clone.hist = hist.Clone(shallow=True)
        return clone


class Channel(_Named, HistFactory.Channel):
    _ROOT = HistFactory.Channel

    def __init__(self, name, samples=None, data=None, inputfile=""):
        # require a name
        super(Channel, self).__init__(name, inputfile)
        if samples is not None:
            for sample in samples:
                self.AddSample(sample)
        if data is not None:
            self.SetData(data)

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

    def sys_names(self):
        """
        Return a list of unique systematic names from OverallSys and HistoSys
        """
        names = []
        for sample in self.samples:
            names.extend(sample.sys_names())
        return list(set(names))

    def sys_hist(self, name=None, where=None):
        """
        Return the effective total low and high histogram for a given
        systematic over samples in this channel.
        If a sample does not contain the named systematic then its nominal
        histogram is used for both low and high variations.

        Parameters
        ----------

        name : string, optional (default=None)
            The systematic name otherwise nominal if None

        where : callable, optional (default=None)
            A callable taking one argument: the sample, and returns True if
            this sample should be included in the total.

        Returns
        -------

        total_low, total_high : histograms
            The total low and high histograms for this systematic

        """
        total_low, total_high = None, None
        for sample in self.samples:
            if where is not None and not where(sample):
                continue
            low, high = sample.sys_hist(name)
            if total_low is None:
                total_low = low.Clone(shallow=True)
            else:
                total_low += low
            if total_high is None:
                total_high = high.Clone(shallow=True)
            else:
                total_high += high
        return total_low, total_high

    def has_sample(self, name):
        for sample in self.samples:
            if sample.name == name:
                return True
        return False

    def has_sample_where(self, func):
        for sample in self.samples:
            if func(sample):
                return True
        return False

    def total(self, where=None, xbin1=1, xbin2=-2):
        """
        Return the total yield and its associated statistical and
        systematic uncertainties.
        """
        nominal, _ = self.sys_hist(None, where=where)
        integral, stat_error = nominal.integral(
            xbin1=xbin1, xbin2=xbin2, error=True)
        ups = [0]
        dns = [0]
        for sys_name in self.sys_names():
            low, high = self.sys_hist(sys_name, where=where)
            up = high.integral(xbin1=xbin1, xbin2=xbin2) - integral
            dn = low.integral(xbin1=xbin1, xbin2=xbin2) - integral
            if up > 0:
                ups.append(up**2)
            else:
                dns.append(up**2)
            if dn > 0:
                ups.append(dn**2)
            else:
                dns.append(dn**2)
        syst_error = (sqrt(sum(ups)), sqrt(sum(dns)))
        return integral, stat_error, syst_error

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

    def RemoveSample(self, name):
        sample_vect = super(Channel, self).GetSamples()
        ivect = sample_vect.begin()
        for sample in sample_vect:
            if sample.GetName() == name:
                sample_vect.erase(ivect)
                break
            ivect.__preinc__()

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

    def apply_snapshot(self, argset):
        """
        Create a clone of this Channel where histograms are modified according
        to the values of the nuisance parameters in the snapshot. This is
        useful when creating post-fit distribution plots.

        Parameters
        ----------

        argset : RooArtSet
            A RooArgSet of RooRealVar nuisance parameters

        Returns
        -------

        channel : Channel
            The modified channel

        """
        clone = self.Clone()
        args = [var for var in argset if not (
            var.name.startswith('binWidth_obs_x_') or
            var.name.startswith('gamma_stat') or
            var.name.startswith('nom_'))]
        # handle NormFactors first
        nargs = []
        for var in args:
            is_norm = False
            name = var.name.replace('alpha_', '')
            for sample in clone.samples:
                if sample.GetNormFactor(name) is not None:
                    log.info("applying snapshot of {0} on sample {1}".format(
                        name, sample.name))
                    is_norm = True
                    # scale the entire sample
                    sample *= var.value
                    # add an OverallSys for the error
                    osys = OverallSys(name,
                        low=1. - var.error / var.value,
                        high=1. + var.error / var.value)
                    sample.AddOverallSys(osys)
                    # remove the NormFactor
                    sample.RemoveNormFactor(name)
            if not is_norm:
                nargs.append(var)
        # modify the nominal shape and systematics
        for sample in clone.samples:
            # check that hist is not NULL
            if sample.hist is None:
                raise RuntimeError(
                    "sample {0} does not have a "
                    "nominal histogram".format(sample.name))
            nominal = sample.hist.Clone(shallow=True)
            for var in nargs:
                name = var.name.replace('alpha_', '')
                if not sample.has_sys(name):
                    continue
                log.info("applying snapshot of {0} on sample {1}".format(
                    name, sample.name))
                low, high = sample.sys_hist(name)
                # modify nominal
                val = var.value
                if val > 0:
                    sample.hist += (high - nominal) * val
                elif val < 0:
                    sample.hist += (nominal - low) * val
                # TODO:
                # modify OverallSys
                # modify HistoSys
        return clone

    def Clone(self):
        clone = Channel(self.name)
        data = self.data
        if data:
            clone.data = data.Clone()
        for sample in self.samples:
            clone.AddSample(sample.Clone())
        clone.hist_path = self.hist_path
        clone.hist_file = self.hist_file
        return clone

    def __iter__(self):
        for sample in super(Channel, self).GetSamples():
            yield asrootpy(sample)

    def __len__(self):
        return len(super(Channel, self).GetSamples())


class Measurement(NamedObject, HistFactory.Measurement):
    _ROOT = HistFactory.Measurement

    def __init__(self, name, channels=None, title=""):
        # require a name
        super(Measurement, self).__init__(name=name, title=title)
        self.SetExportOnly(True)
        if channels is not None:
            for channel in channels:
                self.AddChannel(channel)

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

    def RemoveChannel(self, name):
        channel_vect = super(Measurement, self).GetChannels()
        ivect = channel_vect.begin()
        for channel in channel_vect:
            if channel.GetName() == name:
                channel_vect.erase(ivect)
                break
            ivect.__preinc__()

    def GetChannel(self, name):
        channels = super(Measurement, self).GetChannels()
        for channel in channels:
            if channel.GetName() == name:
                return asrootpy(channel)
        return None

    def GetChannels(self):
        return [asrootpy(c) for c in super(Measurement, self).GetChannels()]

    @property
    def channels(self):
        return self.GetChannels()

    def GetConstantParams(self):
        return list(super(Measurement, self).GetConstantParams())

    @property
    def const_params(self):
        return self.GetConstantParams()

    def Clone(self):
        clone = Measurement(self.name, self.title)
        clone.lumi = self.lumi
        clone.lumi_rel_error = self.lumi_rel_error
        for channel in self.channels:
            clone.AddChannel(channel.Clone())
        for poi in self.GetPOIList():
            clone.AddPOI(poi)
        for const_param in self.const_params:
            clone.AddConstantParam(const_param)
        return clone

    def __iter__(self):
        for channel in super(Measurement, self).GetChannels():
            yield asrootpy(channel)

    def __len__(self):
        return len(super(Measurement, self).GetChannels())
