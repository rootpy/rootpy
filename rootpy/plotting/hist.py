import ROOT
from ..core import Object, isbasictype, camelCaseMethods
from .core import Plottable, dim
from ..objectproxy import *
from ..registry import register
from .style import *
from .graph import *
from array import array
  
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
                if type(nbins) is not int:
                    raise TypeError(
                        "Type of first argument (got %s %s) must be an int" % (type(nbins), nbins))
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
                        "Upper bound (you gave %f) must be greater than lower bound (you gave %f)" % (float(low), float(high)))
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
    
    def underflow(self, axis=1): pass

    def overflow(self, axis=1): pass
    
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
            raise IndexError("bin index %i out of range"% index)
    
    def __setitem__(self, index):
        
        if index not in range(-1, len(self) + 1):
            raise IndexError("bin index %i out of range"% index)

    def __iter__(self):

        return iter(self._content())

    def errors(self):

        return iter(self._error_content())

    def asarray(self):

        return array(self._content())
 
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
                
        self._post_init(**kwargs)
             
    def _post_init(self, **kwargs):
        
        _HistBase.__init__(self)
        self.decorate(**kwargs)
        
        self.xedges = [
            self.GetBinLowEdge(i)
                for i in xrange(1, len(self) + 2)]
        self.xcenters = [
            (self.xedges[i+1] + self.xedges[i])/2
                for i in xrange(len(self)) ]

    def GetMaximum(self, **kwargs):

        return self.maximum(**kwargs)

    def maximum(self, include_error = False):

        if not include_error:
            return self.__class__.__bases__[-1].GetMaximum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i+1, clone.GetBinContent(i+1)+clone.GetBinError(i+1))
        return clone.maximum()
    
    def GetMinimum(self, **kwargs):

        return self.minimum(**kwargs)

    def minimum(self, include_error = False):

        if not include_error:
            return self.__class__.__bases__[-1].GetMinimum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(
                i+1, clone.GetBinContent(i+1)-clone.GetBinError(i+1))
        return clone.minimum()
    
    def expectation(self, startbin = 0, endbin = None):

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

    def yerrors(self):

        for i in xrange(1, self.GetNbinsX()+1):
            yield self.GetBinError(i)
    
    def xerrors(self):

        for edge, center in zip(self.xedges[1:], self.xcenters):
            yield edge - center
    
    def __getitem__(self, index):

        """
        if type(index) is slice:
            return self._content()[index]
        """
        _HistBase.__getitem__(self, index)
        return self.GetBinContent(index+1)

    def __getslice__(self, i, j):

        return self.content()[i,j]
    
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
        
        self._post_init(**kwargs)

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
        
        self._post_init(**kwargs)
            
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
for base1d, base2d, base3d in _HistBase.TYPES.values():
    cls = _Hist_class(rootclass = base1d)
    register()(cls)
    camelCaseMethods(cls)
    cls = _Hist2D_class(rootclass = base2d)
    register()(cls)
    camelCaseMethods(cls)
    cls = _Hist3D_class(rootclass = base3d)
    register()(cls)
    camelCaseMethods(cls)


if ROOT.gROOT.GetVersionCode() >= 334848:

    @camelCaseMethods
    @register()
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

        def errors(self):
            
            for bin in xrange(len(self)):
                yield (self.GetEfficiencyErrorLow(bin+1), self.GetEfficiencyErrorUp(bin+1))

        def GetGraph(self):

            graph = Graph(len(self))
            for index,(bin,effic,(low,up)) in enumerate(zip(xrange(len(self)),iter(self),self.errors())):
                graph.SetPoint(index,self.total.xcenters[bin], effic)
                xerror = (self.total.xedges[bin+1] - self.total.xedges[bin])/2.
                graph.SetPointError(index, xerror, xerror, low, up)
            return graph


class HistStack(Plottable, Object, ROOT.THStack):

    def __init__(self, name = None, title = None, **kwargs):

        Object.__init__(self, name, title)
        self.hists = []
        Plottable.__init__(self)
        self.decorate(**kwargs)
        self.dim = 1
    
    def __dim__(self):

        return self.dim
    
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

        for hist in self.hists:
            yield hist

    def __nonzero__(self):

        return len(self) != 0
    
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

    def lowerbound(self, axis = 1):

        if not self:
            return None # negative infinity
        return min(hist.lowerbound(axis = axis) for hist in self)

    def upperbound(self, axis = 1):
        
        if not self:
            return () # positive infinity
        return max(hist.upperbound(axis = axis) for hist in self)
    
    def GetMaximum(self, **kwargs):

        return self.maximum(**kwargs)

    def maximum(self, **kwargs):

        if not self:
            return None # negative infinity
        return max(hist.maximum(**kwargs) for hist in self)

    def GetMinimum(self, **kwargs):

        return self.minimum(**kwargs)

    def minimum(self, **kwargs):
    
        if not self:
            return () # positive infinity
        return min(hist.minimum(**kwargs) for hist in self)

    def Clone(self, newName = None):

        clone = HistStack(name = newName, title = self.GetTitle())
        clone.decorate(template_object = self)
        for hist in self:
            clone.Add(hist.Clone())
        return clone
    
    def SetLineColor(self, color):
        
        for hist in self:
            hist.SetLineColor(color)
        self.linecolor = convert_color(color, 'mpl')

    def SetLineStyle(self, style):
        
        for hist in self:
            hist.SetLineStyle(style)
        self.linestyle = convert_linestyle(style, 'mpl')

    def SetFillColor(self, color):
        
        for hist in self:
            hist.SetFillColor(color)
        self.fillcolor = convert_color(color, 'mpl')

    def SetFillStyle(self, style):
        
        for hist in self:
            hist.SetFillStyle(style)
        self.fillstyle = convert_fillstyle(style, 'mpl')

    def SetMarkerColor(self, color):
        
        for hist in self:
            hist.SetMarkerColor(color)
        self.markercolor = convert_color(color, 'mpl')

    def SetMarkerStyle(self, style):
        
        for hist in self:
            hist.SetMarkerStyle(style)
        self.markerstyle = convert_markerstyle(style, 'mpl')
