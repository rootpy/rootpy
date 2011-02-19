"""
This module implements python classes which inherit from
and extend the functionality of the ROOT histogram and graph classes.

These histogram classes may be used within other plotting frameworks like
matplotlib while maintaining full compatibility with ROOT.
"""

from operator import add, sub
from array import array
from rootpy.objectproxy import ObjectProxy
from rootpy.core import *
from rootpy.registry import *
import math
import ROOT

try:
    import numpy as array
except:
    import array

class PadMixin(object):

    def Clear(self, *args, **kwargs):

        self.members = []
        self.__class__.__bases__[-1].Clear(self, *args, **kwargs)
    
    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

class Pad(PadMixin, ROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self.members = []
    
class Canvas(PadMixin, ROOT.TCanvas):

    def __init__(self, *args, **kwargs):

        ROOT.TCanvas.__init__(self, *args, **kwargs)
        self.members = []
   
def dim(hist):

    return hist.__class__.DIM

class _HistBase(Plottable, Object):
    
    TYPES = {
        'C': [ROOT.TH1C, ROOT.TH2C, ROOT.TH3C],
        'S': [ROOT.TH1S, ROOT.TH2S, ROOT.TH3S],
        'I': [ROOT.TH1I, ROOT.TH2I, ROOT.TH3I],
        'F': [ROOT.TH1F, ROOT.TH2F, ROOT.TH3F],
        'D': [ROOT.TH1D, ROOT.TH2D, ROOT.TH3D]
    }

    def __init__(self):

        Plottable.__init__(self)
    
    def _parse_args(self, *args):

        params = [{'bins': None,
                   'nbins': None,
                   'low': None,
                   'high': None} for i in xrange(dim(self))]

        for param in params:
            if len(args) == 0:
                raise TypeError("Did not receive expected number of arguments")
            if type(args[0]) in [tuple, list]:
                if list(sorted(args[0])) != list(args[0]):
                    raise ValueError(
                        "Bin edges must be sorted in ascending order")
                if len(set(args[0])) != len(args[0]):
                    raise ValueError("Bin edges must not be repeated")
                param['bins'] = args[0]
                param['nbins'] = len(args[0]) - 1
                args = args[1:]
            elif len(args) >= 3:
                nbins = args[0]
                if not isbasictype(nbins):
                    raise TypeError(
                        "Type of first argument must be int, float, or long")
                low = args[1]
                if not isbasictype(low):
                    raise TypeError(
                        "Type of second argument must be int, float, or long")
                high = args[2]
                if not isbasictype(high):
                    raise TypeError(
                        "Type of third argument must be int, float, or long")
                param['nbins'] = nbins
                param['low'] = low
                param['high'] = high
                if low >= high:
                    raise ValueError(
                        "Upper bound must be greater than lower bound")
                args = args[3:]
            else:
                raise TypeError(
                    "Did not receive expected number of arguments")
        if len(args) != 0:
            raise TypeError(
                "Did not receive expected number of arguments")

        return params

    def Fill(self, *args):

        bin = self.__class__.__bases__[-1].Fill(self, *args)
        if bin > 0:
            return bin - 1
        return bin
    
    def lowerbound(self, axis=1):
        
        if axis == 1:
            return self.xedges[0]
        if axis == 2:
            return self.yedges[0]
        if axis == 3:
            return self.zedges[0]
        return ValueError("axis must be 1, 2, or 3")
    
    def upperbound(self, axis=1):
        
        if axis == 1:
            return self.xedges[-1]
        if axis == 2:
            return self.yedges[-1]
        if axis == 3:
            return self.zedges[-1]
        return ValueError("axis must be 1, 2, or 3")

    def __add__(self, other):
        
        copy = self.Clone()
        copy += other
        return copy
        
    def __iadd__(self, other):
        
        if isbasictype(other):
            if not isinstance(self, _Hist):
                raise ValueError(
                    "A multidimensional histogram must be filled with a tuple")
            self.Fill(other)
        elif type(other) in [list, tuple]:
            if dim(self) not in [len(other), len(other) - 1]:
                raise ValueError(
                    "Dimension of %s does not match dimension "
                    "of histogram (with optional weight as last element)"%
                    str(other))
            self.Fill(*other)
        else:
            self.Add(other)
        return self
    
    def __sub__(self, other):
        
        copy = self.Clone()
        copy -= other
        return copy
        
    def __isub__(self, other):
        
        if isbasictype(other):
            if not isinstance(self, _Hist):
                raise ValueError(
                    "A multidimensional histogram must be filled with a tuple")
            self.Fill(other, -1)
        elif type(other) in [list, tuple]:
            if len(other) == dim(self):
                self.Fill(*(other + (-1, )))
            elif len(other) == dim(self) + 1:
                # negate last element
                self.Fill(*(other[:-1] + (-1 * other[-1], )))
            else:
                raise ValueError(
                    "Dimension of %s does not match dimension "
                    "of histogram (with optional weight as last element)"%
                    str(other))
        else:
            self.Add(other, -1.)
        return self
    
    def __mul__(self, other):
        
        copy = self.Clone()
        copy *= other
        return copy
    
    def __imul__(self, other):
        
        if isbasictype(other):
            self.Scale(other)
            return self
        self.Multiply(other)
        return self
   
    def __div__(self, other):
        
        copy = self.Clone()
        copy /= other
        return copy
    
    def __idiv__(self, other):
        
        if isbasictype(other):
            if other == 0:
                raise ZeroDivisionError()
            self.Scale(1./other)
            return self
        self.Divide(other)
        return self

    def __len__(self):

        return self.GetNbinsX()

    def __getitem__(self, index):

        if index not in range(-1, len(self) + 1):
            raise IndexError("bin index out of range")
    
    def __setitem__(self, index):

        if index not in range(-1, len(self) + 1):
            raise IndexError("bin index out of range")

    def __iter__(self):

        return iter(self._content())

    def itererrors(self):

        return iter(self._error_content())

    def asarray(self):

        return array.array(self._content())
 
class _Hist(_HistBase):
    
    DIM = 1
        
    def __init__(self, *args, **kwargs):
                
        name = kwargs.get('name', None)
        title = kwargs.get('title', None)
        
        params = self._parse_args(*args)
        
        if params[0]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'])
        else:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']))
                
        self._post_init()
             
    def _post_init(self, **kwargs):
        
        _HistBase.__init__(self)
        self.decorate(**kwargs)

        self.xedges = [
            self.GetBinLowEdge(i)
                for i in xrange(1, len(self) + 2)]
        self.xcenters = [
            (self.xedges[i+1] + self.xedges[i])/2
                for i in xrange(len(self)) ]

    def GetMaximum(self, include_error = False):

        if not include_error:
            return ROOT.TH1F.GetMaximum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i+1, clone.GetBinContent(i+1)+clone.GetBinError(i+1))
        return clone.GetMaximum()
    
    def GetMinimum(self, include_error = False):

        if not include_error:
            return ROOT.TH1F.GetMinimum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i+1, clone.GetBinContent(i+1)-clone.GetBinError(i+1))
        return clone.GetMinimum()
    
    def Expectation(self, startbin = 0, endbin = None):

        if endbin is not None and endbin < startbin:
            raise DomainError("endbin should be greated than startbin")
        if endbin is None:
            endbin = len(self)-1
        expect = 0.
        norm = 0.
        for index in xrange(startbin, endbin+1):
            val = self[index]
            expect += val * self.xcenters[index]
            norm += val
        return expect / norm if norm > 0 else (self.xedges[endbin+1] + self.xedges[startbin])/2
     
    def _content(self):

        return [self.GetBinContent(i) for i in xrange(1, self.GetNbinsX()+1)]
    
    def _error_content(self):

        return [self.GetBinError(i) for i in xrange(1, self.GetNbinsX()+1)]

    def __getitem__(self, index):

        _HistBase.__getitem__(self, index)
        return self.GetBinContent(index+1)
    
    def __setitem__(self, index, value):

        _HistBase.__setitem__(self, index)
        self.SetBinContent(index+1, value)

class _Hist2D(_HistBase):
    
    DIM = 2

    def __init__(self, *args, **kwargs):
        
        name = kwargs.get('name', None)
        title = kwargs.get('title', None)
        
        params = self._parse_args(*args)
        
        if params[0]['bins'] is None and params[1]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'])
        elif params[0]['bins'] is None and params[1]['bins'] is not None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], array('d', params[1]['bins']))
        elif params[0]['bins'] is not None and params[1]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], params[1]['low'], params[1]['high'])
        else:
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']))
        
        self._post_init()

    def _post_init(self, **kwargs):

        _HistBase.__init__(self)
        self.decorate(**kwargs)
         
        self.xedges = [
            self.GetXaxis().GetBinLowEdge(i)
                for i in xrange(1, len(self) + 2)]
        self.xcenters = [
            (self.xedges[i+1] + self.xedges[i])/2
                for i in xrange(len(self))]
        self.yedges = [
            self.GetYaxis().GetBinLowEdge(i)
                for i in xrange(1, len(self[0]) + 2)]
        self.ycenters = [
            (self.yedges[i+1] + self.yedges[i])/2
                for i in xrange(len(self[0]))]

    def _content(self):

        return [[
            self.GetBinContent(i, j)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]
    
    def _error_content(self):

        return [[
            self.GetBinError(i, j)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]

    def __getitem__(self, index):
        
        _HistBase.__getitem__(self, index)
        a = ObjectProxy([
            self.GetBinContent(index+1, j)
                for j in xrange(1, self.GetNbinsY() + 1)])
        a.__setposthook__('__setitem__', self._setitem(index))
        return a
    
    def _setitem(self, i):
        def __setitem(j, value):
            self.SetBinContent(i+1, j+1, value)
        return __setitem

class _Hist3D(_HistBase):

    DIM = 3

    def __init__(self, *args, **kwargs):

        name = kwargs.get('name', None)
        title = kwargs.get('title', None)
        
        params = self._parse_args(*args)

        # ROOT is missing constructors for TH3F...
        if params[0]['bins'] is None and \
           params[1]['bins'] is None and \
           params[2]['bins'] is None:
            Object.__init__(self, name, title,
                params[0]['nbins'], params[0]['low'], params[0]['high'],
                params[1]['nbins'], params[1]['low'], params[1]['high'],
                params[2]['nbins'], params[2]['low'], params[2]['high'])
        else:
            if params[0]['bins'] is None:
                step = (params[0]['high'] - params[0]['low'])\
                    / float(params[0]['nbins'])
                params[0]['bins'] = [
                    params[0]['low'] + n*step
                        for n in xrange(params[0]['nbins'] + 1)]
            if params[1]['bins'] is None:
                step = (params[1]['high'] - params[1]['low'])\
                    / float(params[1]['nbins'])
                params[1]['bins'] = [
                    params[1]['low'] + n*step
                        for n in xrange(params[1]['nbins'] + 1)]
            if params[2]['bins'] is None:
                step = (params[2]['high'] - params[2]['low'])\
                    / float(params[2]['nbins'])
                params[2]['bins'] = [
                    params[2]['low'] + n*step
                        for n in xrange(params[2]['nbins'] + 1)]
            Object.__init__(self, name, title,
                params[0]['nbins'], array('d', params[0]['bins']),
                params[1]['nbins'], array('d', params[1]['bins']),
                params[2]['nbins'], array('d', params[2]['bins']))
        
        self._post_init()
            
    def _post_init(self, **kwargs):
        
        _HistBase.__init__(self)
        self.decorate(**kwargs)

        self.xedges = [
            self.GetXaxis().GetBinLowEdge(i)
                for i in xrange(1, len(self) + 2)]
        self.xcenters = [
            (self.xedges[i+1] + self.xedges[i])/2
                for i in xrange(len(self))]
        self.yedges = [
            self.GetYaxis().GetBinLowEdge(i)
                for i in xrange(1, len(self[0]) + 2)]
        self.ycenters = [
            (self.yedges[i+1] + self.yedges[i])/2
                for i in xrange(len(self[0]))]
        self.zedges = [
            self.GetZaxis().GetBinLowEdge(i)
                for i in xrange(1, len(self[0][0]) + 2)]
        self.zcenters = [
            (self.zedges[i+1] + self.zedges[i])/2
                for i in xrange(len(self[0][0]))]
    
    def _content(self):

        return [[[
            self.GetBinContent(i, j, k)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]
                        for k in xrange(1, self.GetNbinsZ() + 1)]
    
    def _error_content(self):

        return [[[
            self.GetBinError(i, j, k)
                for i in xrange(1, self.GetNbinsX() + 1)]
                    for j in xrange(1, self.GetNbinsY() + 1)]
                        for k in xrange(1, self.GetNbinsZ() + 1)]

    def __getitem__(self, index):
        
        _HistBase.__getitem__(self, index)
        out = []
        for j in xrange(1, self.GetNbinsY() + 1):
            a = ObjectProxy([
                self.GetBinContent(index+1, j, k)
                    for k in xrange(1, self.GetNbinsZ() + 1)])
            a.__setposthook__('__setitem__', self._setitem(index, j-1))
            out.append(a)
        return out
    
    def _setitem(self, i, j):
        def __setitem(k, value):
            self.SetBinContent(i+1, j+1, k+1, value)
        return __setitem

def _Hist_class(bintype = 'F', rootclass = None):

    if rootclass is None:
        bintype = bintype.upper()
        if not _HistBase.TYPES.has_key(bintype):
            raise TypeError("No histogram available with bintype %s"% bintype)
        rootclass = _HistBase.TYPES[bintype][0]
    class Hist(_Hist, rootclass): pass
    return Hist

def _Hist2D_class(bintype = 'F', rootclass = None):

    if rootclass is None:
        bintype = bintype.upper()
        if not _HistBase.TYPES.has_key(bintype):
            raise TypeError("No histogram available with bintype %s"% bintype)
        rootclass = _HistBase.TYPES[bintype][1]
    class Hist2D(_Hist2D, rootclass): pass
    return Hist2D

def _Hist3D_class(bintype = 'F', rootclass = None):
    
    if rootclass is None:
        bintype = bintype.upper()
        if not _HistBase.TYPES.has_key(bintype):
            raise TypeError("No histogram available with bintype %s"% bintype)
        rootclass = _HistBase.TYPES[bintype][2]
    class Hist3D(_Hist3D, rootclass): pass
    return Hist3D

def Hist(*args, **kwargs):
    """
    Returns a 1-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the bintype keyword argument)
    """
    return _Hist_class(bintype = kwargs.get('bintype','F'))(*args, **kwargs)

def Hist2D(*args, **kwargs):
    """
    Returns a 2-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the bintype keyword argument)
    """
    return _Hist2D_class(bintype = kwargs.get('bintype','F'))(*args, **kwargs)
   
def Hist3D(*args, **kwargs):
    """
    Returns a 3-dimensional Hist object which inherits from the associated
    ROOT.TH1* class (where * is C, S, I, F, or D depending on the bintype keyword argument)
    """
    return _Hist3D_class(bintype = kwargs.get('bintype','F'))(*args, **kwargs)

# register the classes
for value in _HistBase.TYPES.values():
    cls = _Hist_class(rootclass = value[0])
    register(cls, cls._post_init)
    cls = _Hist2D_class(rootclass = value[1])
    register(cls, cls._post_init)
    cls = _Hist3D_class(rootclass = value[2])
    register(cls, cls._post_init)

class Efficiency(Plottable, Object, ROOT.TEfficiency):

    def __init__(self, passed, total, name = None, title = None, **kwargs):

        if dim(passed) != 1 or dim(total) != 1:
            raise TypeError("histograms must be 1 dimensional")
        if len(passed) != len(total):
            raise ValueError("histograms must have the same number of bins")
        if passed.xedges != total.xedges:
            raise ValueError("histograms do not have the same bin boundaries")
        Object.__init__(self, name, title, len(total), total.xedges[0], total.xedges[-1])
        self.passed = passed.Clone()
        self.total = total.Clone()
        self.SetPassedHistogram(self.passed, 'f')
        self.SetTotalHistogram(self.total, 'f') 
        Plottable.__init__(self)
        self.decorate(**kwargs)
    
    def __len__(self):
    
        return len(self.total)

    def __getitem__(self, bin):

        return self.GetEfficiency(bin+1)
    
    def __add__(self, other):

        copy = self.Clone()
        copy.Add(other)
        return copy

    def __iadd__(self, other):

        ROOT.TEfficiency.Add(self, other)
        return self

    def __iter__(self):

        for bin in xrange(len(self)):
            yield self[bin]

    def itererrors(self):
        
        for bin in xrange(len(self)):
            yield (self.GetEfficiencyErrorLow(bin+1), self.GetEfficiencyErrorUp(bin+1))

    def GetGraph(self):

        graph = Graph(len(self))
        index = 0
        for bin,effic,(low,up) in zip(xrange(len(self)),iter(self),self.itererrors()):
            if effic > 0:
                graph.SetPoint(index,self.total.xcenters[bin], effic)
                xerror = (self.total.xedges[bin+1] - self.total.xedges[bin])/2.
                graph.SetPointError(index, xerror, xerror, low, up)
                index += 1
        graph.Set(index)
        return graph

class Graph(Plottable, NamelessConstructorObject, ROOT.TGraphAsymmErrors):

    def __init__(self, npoints = 0, hist = None, efficiency = None, file = None, name = None, title = None, **kwargs):

        if hist is not None:
            NamelessConstructorObject.__init__(self, name, title, hist)
        elif npoints > 0:
            NamelessConstructorObject.__init__(self, name, title, npoints)
        elif isinstance(file, basestring):
            gfile = open(file, 'r')
            lines = gfile.readlines()
            gfile.close()
            NamelessConstructorObject.__init__(self, name, title, len(lines)+2)
            pointIndex = 0
            for line in lines:
                try:
                    X, Y = [float(s) for s in line.strip(" //").split()]
                    self.SetPoint(pointIndex, X, Y)
                    pointIndex += 1
                except: pass
            self.Set(pointIndex)
        else:
            raise ValueError()

        Plottable.__init__(self)
        self.decorate(**kwargs)
    
    def __dim__(self):

        return 1
    
    def __len__(self):
    
        return self.GetN()

    def __getitem__(self, index):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        return (self.GetX()[index], self.GetY()[index])

    def __setitem__(self, index, point):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        if type(point) not in [list, tuple]:
            raise TypeError("argument must be a tuple or list")
        if len(point) != 2:
            raise ValueError("argument must be of length 2")
        self.SetPoint(index, point[0], point[1])
    
    def __iter__(self):

        for index in xrange(len(self)):
            yield self[index]
    
    def iterx(self):

        x = self.GetX()
        for index in xrange(len(self)):
            yield x[index]

    def itery(self):
        
        y = self.GetY()
        for index in xrange(len(self)):
            yield y[index]

    def itererrorsx(self):
        
        high = self.GetEXhigh()
        low = self.GetEXlow()
        for index in xrange(len(self)):
            yield (low[index], high[index])
    
    def itererrorsxhigh(self):
        
        high = self.GetEXhigh()
        for index in xrange(len(self)):
            yield high[index]

    def itererrorsxlow(self):
        
        low = self.GetEXlow()
        for index in xrange(len(self)):
            yield low[index]

    def itererrorsy(self):
        
        high = self.GetEYhigh()
        low = self.GetEYlow()
        for index in xrange(len(self)):
            yield (low[index], high[index])
    
    def itererrorsyhigh(self):
        
        high = self.GetEYhigh()
        for index in xrange(len(self)):
            yield high[index]
    
    def itererrorsylow(self):
        
        low = self.GetEYlow()
        for index in xrange(len(self)):
            yield low[index]

    def __add__(self, other):

        copy = self.Clone()
        copy += other
        return copy

    def __iadd__(self, other):
        
        if len(other) != len(self):
            raise ValueError("graphs do not contain the same number of points")
        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] + other)
        else:
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = math.sqrt((self.GetEYlow()[index])**2 + (other.GetEYlow()[index])**2)
                yhigh = math.sqrt((self.GetEYhigh()[index])**2 + (other.GetEYhigh()[index])**2)
                self.SetPoint(index, mypoint[0], mypoint[1]+otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __sub__(self, other):

        copy = self.Clone()
        copy -= other
        return copy

    def __isub__(self, other):
        
        if len(other) != len(self):
            raise ValueError("graphs do not contain the same number of points")
        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] - other)
        else:
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = math.sqrt((self.GetEYlow()[index])**2 + (other.GetEYlow()[index])**2)
                yhigh = math.sqrt((self.GetEYhigh()[index])**2 + (other.GetEYhigh()[index])**2)
                self.SetPoint(index, mypoint[0], mypoint[1]-otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __div__(self, other):

        copy = other.Clone()
        copy /= other
        return copy

    def __idiv__(self, other):
        
        if len(other) != len(self):
            raise ValueError("graphs do not contain the same number of points")
        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index] 
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1]/other)
                self.SetPointError(index, xlow, xhigh, ylow/other, yhigh/other)
        else:
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = (mypoint[1]/otherpoint[1])*math.sqrt((self.GetEYlow()[index]/mypoint[1])**2 + (other.GetEYlow()[index]/otherpoint[1])**2)
                yhigh = (mypoint[1]/otherpoint[1])*math.sqrt((self.GetEYhigh()[index]/mypoint[1])**2 + (other.GetEYhigh()[index]/otherpoint[1])**2)
                self.SetPoint(index, mypoint[0], mypoint[1]/otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __mul__(self, other):

        copy = self.Clone()
        copy *= other
        return copy

    def __imul__(self, other):
        
        if len(other) != len(self):
            raise ValueError("graphs do not contain the same number of points")
        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index] 
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1]*other)
                self.SetPointError(index, xlow, xhigh, ylow*other, yhigh*other)
        else:
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = (mypoint[1]*otherpoint[1])*math.sqrt((self.GetEYlow()[index]/mypoint[1])**2 + (other.GetEYlow()[index]/otherpoint[1])**2)
                yhigh = (mypoint[1]*otherpoint[1])*math.sqrt((self.GetEYhigh()[index]/mypoint[1])**2 + (other.GetEYhigh()[index]/otherpoint[1])**2)
                self.SetPoint(index, mypoint[0], mypoint[1]*otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self
     
    def setErrorsFromHist(self, hist):

        if hist.GetNbinsX() != self.GetN(): return
        for i in range(hist.GetNbinsX()):
            content = hist.GetBinContent(i+1)
            if content > 0:
                self.SetPointEYhigh(i, content)
                self.SetPointEYlow(i, 0.)
            else:
                self.SetPointEYlow(i, -1*content)
                self.SetPointEYhigh(i, 0.)

    def GetMaximum(self, include_error = False):

        if not include_error:
            return self.yMax()
        summed = map(add, self.itery(), self.itererrorsyhigh())
        return max(summed)

    def GetMinimum(self, include_error = False):

        if not include_error:
            return self.yMin()
        summed = map(sub, self.itery(), self.itererrorsylow())
        return min(summed)
    
    def xMin(self):
        
        if len(self.getX()) == 0:
            raise ValueError("Can't get xmin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(), self.GetX())
    
    def xMax(self):

        if len(self.getX()) == 0:
            raise ValueError("Can't get xmax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetX())

    def yMin(self):
        
        if len(self.getY()) == 0:
            raise ValueError("Can't get ymin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(), self.GetY())

    def yMax(self):
    
        if len(self.getY()) == 0:
            raise ValueError("Can't get ymax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetY())

    def Crop(self, x1, x2, copy = False):

        numPoints = self.GetN()
        if copy:
            cropGraph = self.Clone()
            copyGraph = self
        else:
            cropGraph = self
            copyGraph = self.Clone()
        X = copyGraph.GetX()
        EXlow = copyGraph.GetEXlow()
        EXhigh = copyGraph.GetEXhigh()
        Y = copyGraph.GetY()
        EYlow = copyGraph.GetEYlow()
        EYhigh = copyGraph.GetEYhigh()
        xmin = copyGraph.xMin()
        if x1 < xmin:
            cropGraph.Set(numPoints+1)
            numPoints += 1
        xmax = copyGraph.xMax()
        if x2 > xmax:
            cropGraph.Set(numPoints+1)
            numPoints += 1
        index = 0
        for i in xrange(numPoints):
            if i == 0 and x1 < xmin:
                cropGraph.SetPoint(0, x1, copyGraph.Eval(x1))
            elif i == numPoints - 1 and x2 > xmax:
                cropGraph.SetPoint(i, x2, copyGraph.Eval(x2))
            else:
                cropGraph.SetPoint(i, X[index], Y[index])
                cropGraph.SetPointError(i, EXlow[index], EXhigh[index],
                                           EYlow[index], EYhigh[index])
                index += 1
        return cropGraph

    def Reverse(self, copy = False):
        
        numPoints = self.GetN()
        if copy:
            revGraph = self.Clone()
        else:
            revGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in xrange(numPoints):
            index = numPoints-1-i
            revGraph.SetPoint(i, X[index], Y[index])
            revGraph.SetPointError(i, EXlow[index], EXhigh[index],
                                      EYlow[index], EYhigh[index])
        return revGraph
         
    def Invert(self, copy = False):

        numPoints = self.GetN()
        if copy:
            invGraph = self.Clone()
        else:
            invGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in xrange(numPoints):
            invGraph.SetPoint(i, Y[i], X[i])
            invGraph.SetPointError(i, EYlow[i], EYhigh[i],
                                      EXlow[i], EXhigh[i])
        return invGraph
 
    def Scale(self, value, copy = False):

        xmin, xmax = self.GetXaxis().GetXmin(), self.GetXaxis().GetXmax()
        numPoints = self.GetN()
        if copy:
            scaleGraph = self.Clone()
        else:
            scaleGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in xrange(numPoints):
            scaleGraph.SetPoint(i, X[i], Y[i]*value)
            scaleGraph.SetPointError(i, EXlow[i], EXhigh[i],
                                        EYlow[i]*value, EYhigh[i]*value)
        scaleGraph.GetXaxis().SetLimits(xmin, xmax)
        scaleGraph.GetXaxis().SetRangeUser(xmin, xmax)
        scaleGraph.integral = self.integral * value
        return scaleGraph

    def Stretch(self, value, copy = False):

        numPoints = self.GetN()
        if copy:
            stretchGraph = self.Clone()
        else:
            stretchGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in xrange(numPoints):
            stretchGraph.SetPoint(i, X[i]*value, Y[i])
            stretchGraph.SetPointError(i, EXlow[i]*value, EXhigh[i]*value,
                                          EYlow[i], EYhigh[i])
        return stretchGraph
    
    def Shift(self, value, copy = False):

        numPoints = self.GetN()
        if copy:
            shiftGraph = self.Clone()
        else:
            shiftGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in xrange(numPoints):
            shiftGraph.SetPoint(i, X[i]+value, Y[i])
            shiftGraph.SetPointError(i, EXlow[i], EXhigh[i],
                                        EYlow[i], EYhigh[i])
        return shiftGraph
        
    def Integrate(self):
    
        area = 0.
        X = self.GetX()
        Y = self.GetY()
        for i in xrange(self.GetN()-1):
            area += (X[i+1] - X[i])*(Y[i] + Y[i+1])/2.
        return area

register(Graph)

class HistStack(Plottable, Object, ROOT.THStack):

    DIM = 1
    
    def __init__(self, name = None, title = None, **kwargs):

        Object.__init__(self, name, title)
        self.hists = []
        Plottable.__init__(self)
        self.decorate(**kwargs)
        self.dim = 1
    
    def GetHists(self):

        return [hist for hist in self.hists]
    
    def Add(self, hist):

        if isinstance(hist, _Hist) or isinstance(hist, _Hist2D):
            if not self:
                self.dim = dim(hist)
            elif dim(self) != dim(hist):
                raise TypeError("Dimension of histogram does not match dimension of already contained histograms")
            if hist not in self:
                self.hists.append(hist)
                ROOT.THStack.Add(self, hist, hist.format)
        else:
            raise TypeError("Only 1D and 2D histograms are supported")
    
    def get_sum(self):
        """
        Return a histogram which is the sum of all histgrams in the stack
        """
        if not self:
            return None
        hist_template = self[0].Clone()
        for hist in self[1:]:
            hist_template += hist
        return hist_template
    
    def __add__(self, other):

        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and %s"%
                other.__class__.__name__)
        clone = HistStack()
        for hist in self:
            clone.Add(hist)
        for hist in other:
            clone.Add(hist)
        return clone
    
    def __iadd__(self, other):
        
        if not isinstance(other, HistStack):
            raise TypeError(
                "Addition not supported for HistStack and %s"%
                other.__class__.__name__)
        for hist in other:
            self.Add(hist)
        return self

    def __len__(self):

        return len(self.GetHists())
    
    def __getitem__(self, index):

        return self.GetHists()[index]

    def __iter__(self):

        return iter(self.GetHists())

    def __nonzero__(self):

        return len(self) == 0
    
    def Scale(self, value):

        for hist in self:
            hist.Scale(value)

    def Integral(self, start = None, end = None):
        
        integral = 0
        if start != None and end != None:
            for hist in self:
                integral += hist.Integral(start, end)
        else:
            for hist in self:
                integral += hist.Integral()
        return integral

    def GetMaximum(self, include_error = False):

        _max = None # negative infinity
        for hist in self:
            lmax = hist.GetMaximum(include_error = include_error)
            if lmax > _max:
                _max = lmax
        return _max

    def GetMinimum(self, include_error = False):

        _min = () # positive infinity
        for hist in self:
            lmin = hist.GetMinimum(include_error = include_error)
            if lmin < _min:
                _min = lmin
        return _min

    def Clone(self, newName = None):

        clone = HistStack(name = newName, title = self.GetTitle())
        clone.decorate(template_object = self)
        for hist in self:
            clone.Add(hist.Clone())
        return clone
    
    def SetLineColor(self, color):

        if colors.has_key(color):
            for hist in self:
                hist.SetLineColor(colors[color])
            self.linecolor = color
        elif color in colors.values():
            for hist in self:
                hist.SetLineColor(color)
            self.linecolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def SetLineStyle(self, style):
        
        if lines.has_key(style):
            for hist in self:
                hist.SetLineStyle(lines[style])
            self.linestyle = style
        elif style in lines.values():
            for hist in self:
                hist.SetLineStyle(style)
            self.linestyle = style
        else:
            raise ValueError("Line style %s not understood"% style)

    def SetFillColor(self, color):
        
        if colors.has_key(color):
            for hist in self:
                hist.SetFillColor(colors[color])
            self.fillcolor = color
        elif color in colors.values():
            for hist in self:
                hist.SetFillColor(color)
            self.fillcolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def SetFillStyle(self, style):
        
        if fills.has_key(style):
            for hist in self:
                hist.SetFillStyle(fills[style])
            self.fillstyle = style
        elif style in fills.values():
            for hist in self:
                hist.SetFillStyle(style)
            self.fillstyle = style
        else:
            raise ValueError("Fill style %s not understood"% style)

    def SetMarkerColor(self, color):
        
        if colors.has_key(color):
            for hist in self:
                hist.SetMarkerColor(colors[color])
            self.markercolor = color
        elif color in colors.values():
            for hist in self:
                hist.SetMarkerColor(color)
            self.markercolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def SetMarkerStyle(self, style):
        
        if markers.has_key(style):
            for hist in self:
                hist.SetFillStyle(markers[style])
            self.markerstyle = style
        elif style in markers.values():
            for hist in self:
                hist.SetFillStyle(style)
            self.markerstyle = style
        else:
            raise ValueError("Marker style %s not understood"% style)


class Legend(Object, ROOT.TLegend):

    def __init__(self, nentries, pad,
                       leftmargin = 0.,
                       textfont = None,
                       textsize = 0.04,
                       fudge = 1.):
   
        buffer = 0.03
        height = fudge * 0.04 * nentries + buffer
        ROOT.TLegend.__init__(self, pad.GetLeftMargin() + buffer + leftmargin,
                                    (1. - pad.GetTopMargin()) - height,
                                    1. - pad.GetRightMargin(),
                                    ((1. - pad.GetTopMargin()) - buffer))
        self.UseCurrentStyle()
        self.SetEntrySeparation(0.2)
        self.SetMargin(0.15)
        self.SetFillStyle(0)
        self.SetFillColor(0)
        if textfont:
            self.SetTextFont(textfont)
        self.SetTextSize(textsize)
        self.SetBorderSize(0)

    def Height(self):
        
        return abs(self.GetY2() - self.GetY1())

    def Width(self):

        return abs(self.GetX2() - self.GetX1())
    
    def AddEntry(self, object):

        if isinstance(object, HistStack):
            for hist in object:
                if object.inlegend:
                    ROOT.TLegend.AddEntry(self, hist, hist.GetTitle(), object.legendstyle)
        elif isinstance(object, Plottable):
            if object.inlegend:
                ROOT.TLegend.AddEntry(self, object, object.GetTitle(), object.legendstyle)
        else:
            raise TypeError("Can't add object of type %s to legend"%\
                type(object))
