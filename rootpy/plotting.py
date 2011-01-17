from operator import add, sub
import ROOT
from array import array
from style import markers, colors, lines, fills
from objectproxy import ObjectProxy
import uuid

def asrootpy(tobject):

    if issubclass(tobject.__class__, ROOT.TH1D):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = Hist1D
        tobject.decorate(template)
    elif issubclass(tobject.__class__, ROOT.TH2D):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = Hist2D
        tobject.decorate(template)
    elif issubclass(tobject.__class__, ROOT.TH3D):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = Hist3D
        tobject.decorate(template)
    elif issubclass(tobject.__class__, ROOT.TGraphAsymmErrors):
        template = _Plottable()
        template.decorate(tobject)
        tobject.__class__ = Graph
        tobject.decorate(template)
    return tobject

class _Object(object):

    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = uuid.uuid4().hex
        if isinstance(self, Graph):
            self.__class__.__bases__[-1].__init__(self, *args, **kwargs)
            self.SetName(name)
            self.SetTitle(title)
        else:
            self.__class__.__bases__[-1].__init__(self, name, title, *args, **kwargs)

    def Clone(self, newName = None):

        if newName:
            clone = self.__class__.__bases__[-1].Clone(self, newName)
        else:
            clone = self.__class__.__bases__[-1].Clone(self, self.GetName()+'_clone')
        clone.__class__ = self.__class__
        if issubclass(self.__class__, _Plottable):
            clone.decorate(self)
        return clone

    def __copy__(self):

        return self.Clone()

    def __deepcopy__(self, memo):

        return self.Clone()

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "%s(%s)"%(self.__class__.__name__, self.GetTitle())

class Legend(_Object, ROOT.TLegend):

    def __init__(self, nentries, pad, leftmargin = 0., textfont = None, textsize = 0.04, fudge = 1.):
   
        buffer = 0.03
        height = fudge*0.04*numEntries + buffer
        ROOT.TLegend.__init__(self,pad.GetLeftMargin()+buffer+leftmargin,(1.-pad.GetTopMargin()) - height,1.-pad.GetRightMargin(),((1.-pad.GetTopMargin())-buffer))
        legend.UseCurrentStyle()
        legend.SetEntrySeparation(0.2)
        legend.SetMargin(0.15)
        legend.SetFillStyle(0)
        legend.SetFillColor(0)
        if textfont:
            legend.SetTextFont(textfont)
        legend.SetTextSize(textsize)
        legend.SetBorderSize(0)

    def Height(self):
        
        return abs(self.GetY2() - self.GetY1())

    def Width(self):

        return abs(self.GetX2() - self.GetX1())
    
    def AddEntry(self, object):

        if isinstance(object, HistStack):
            for hist in object:
                if hist.inlegend:
                    ROOT.TLegend.AddEntry(self, hist, hist.GetTitle(), hist.legendstyle)
        elif issubclass(object.__class__, _Plottable):
            if object.inlegend:
                ROOT.TLegend.AddEntry(self, object, object.GetTitle(), object.legendstyle)
        else:
            raise TypeError("Can't add object of type %s to legend"% type(object))

class _Plottable(object):

    def decorate(self, template_object = None, **kwargs):
        
        self.norm  = kwargs.get('norm', None)
        self.format = kwargs.get('format', "EP")
        self.legendstyle = kwargs.get('legendstyle', "P")
        self.intMode = kwargs.get('intMode', False)
        self.visible = kwargs.get('visible', True)
        self.inlegend = kwargs.get('inLegend', True)
        self.markerstyle = kwargs.get('markerstyle', "circle")
        self.markercolor = kwargs.get('markercolor', "black")
        self.fillcolor = kwargs.get('fillcolor', "white")
        self.fillstyle = kwargs.get('fillstyle', "hollow")
        self.linecolor = kwargs.get('linecolor', "black")
        self.linestyle = kwargs.get('linestyle', "")

        if issubclass(template_object.__class__, _Plottable):
            self.decorate(**template_object.__decorators())
        else:
            if issubclass(template_object.__class__, ROOT.TAttLine):
                self.linecolor = template_object.GetLineColor()
                self.linestyle = template_object.GetLineStyle()
            if issubclass(template_object.__class__, ROOT.TAttFill):
                self.fillcolor = template_object.GetFillColor()
                self.fillstyle = template_object.GetFillStyle()
            if issubclass(template_object.__class__, ROOT.TAttMarker):
                self.markercolor = template_object.GetMarkerColor()
                self.markerstyle = template_object.GetMarkerStyle()
       
        if issubclass(self.__class__, ROOT.TAttFill):
            if self.fillcolor not in ["white", ""] and self.fillstyle not in ["", "hollow"]:
                self.SetFillStyle(self.fillstyle)
            else:
                self.SetFillStyle("solid")
            self.SetFillColor(self.fillcolor)
        if issubclass(self.__class__, ROOT.TAttLine):
            self.SetLineStyle(self.linestyle)
            self.SetLineColor(self.linecolor)
        if issubclass(self.__class__, ROOT.TAttMarker):
            self.SetMarkerStyle(self.markerstyle)
            self.SetMarkerColor(self.markercolor)
     
    def __decorators(self):
    
        return {
            "format"        : self.format,
            "legendstyle"   : self.legendstyle,
            "intMode"       : self.intMode,
            "visible"       : self.visible,
            "inlegend"      : self.inlegend,
            "markercolor"  : self.markercolor,
            "markerstyle"   : self.markerstyle,
            "fillcolor"    : self.fillcolor,
            "fillstyle"     : self.fillstyle,
            "linecolor"    : self.linecolor,
            "linestyle"     : self.linestyle
        }

    def SetLineColor(self, color):

        if colors.has_key(color):
            self.__class__.__bases__[-1].SetLineColor(self, colors[color])
        elif color in colors.values():
            self.__class__.__bases__[-1].SetLineColor(self, color)
        else:
            raise ValueError("Color %s not understood"% color)

    def SetLineStyle(self, style):
        
        if lines.has_key(style):
            self.__class__.__bases__[-1].SetLineStyle(self, lines[style])
        elif style in lines.values():
            self.__class__.__bases__[-1].SetLineStyle(self, style)
        else:
            raise ValueError("Line style %s not understood"% style)

    def SetFillColor(self, color):
        
        if colors.has_key(color):
            self.__class__.__bases__[-1].SetFillColor(self, colors[color])
        elif color in colors.values():
            self.__class__.__bases__[-1].SetFillColor(self, color)
        else:
            raise ValueError("Color %s not understood"% color)

    def SetFillStyle(self, style):
        
        if fills.has_key(style):
            self.__class__.__bases__[-1].SetFillStyle(self, fills[style])
        elif style in fills.values():
            self.__class__.__bases__[-1].SetFillStyle(self, style)
        else:
            raise ValueError("Fill style %s not understood"% style)

    def SetMarkerColor(self, color):
        
        if colors.has_key(color):
            self.__class__.__bases__[-1].SetMarkerColor(self, colors[color])
        elif color in colors.values():
            self.__class__.__bases__[-1].SetMarkerColor(self, color)
        else:
            raise ValueError("Color %s not understood"% color)

    def SetMarkerStyle(self, style):
        
        if markers.has_key(style):
            self.__class__.__bases__[-1].SetFillStyle(self, markers[style])
        elif style in markers.values():
            self.__class__.__bases__[-1].SetFillStyle(self, style)
        else:
            raise ValueError("Marker style %s not understood"% style)

    def Draw(self, *args):
                
        if self.visible:
            if self.format:
                self.__class__.__bases__[-1].Draw(self, " ".join((self.format,)+args))
            else:
                self.__class__.__bases__[-1].Draw(self, " ".join(args))

class _HistBase(_Plottable, _Object):
     
    def __add__(self, other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other)
        return copy
        
    def __sub__(self, other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other, -1.)
        return copy
        
    def __mul__(self, other):
        
        copy = self.Clone(self.GetName()+"_clone")
        if type(other) in [float, int]:
            copy.Scale(other)
            return copy
        copy.Multiply(other)
        return copy
        
    def __div__(self, other):
        
        copy = self.Clone(self.GetName()+"_clone")
        if type(other) in [float, int]:
            if other == 0:
                raise ZeroDivisionError()
            copy.Scale(1./other)
            return copy
        copy.Divide(other)
        return copy
    
    def __len__(self):

        return self.GetNbinsX()

    def __getitem__(self, index):

        if index not in range(-1, self.GetNbinsX()+1):
            raise IndexError("bin index out of range")
    
    def __setitem__(self, index):

        if index not in range(-1, self.GetNbinsX()+1):
            raise IndexError("bin index out of range")

    def __iter__(self):

        return iter(self._content())

class HistStack(_Object, ROOT.THStack):

    def __init__(self, name = None, title = None, norm = None):

        _Object.__init__(self, name, title)
        self.norm = norm
        self.hists = []

    def GetHists(self):

        return [hist for hist in self.hists]
    
    def Add(self, hist):

        if isinstance(hist, Hist1D) or isinstance(hist, Hist2D):
            if hist not in self:
                self.hists.append(hist)
                ROOT.THStack.Add(self, hist, hist.format)
        else:
            raise TypeError("Only 1D and 2D histograms are supported")
    
    def __add__(self, other):

        if not isinstance(other, HistStack):
            raise TypeError("Addition not supported for HistStack and %s"% other.__class__.__name__)
        clone = HistStack()
        for hist in self:
            clone.Add(hist)
        for hist in other:
            clone.Add(hist)
        return clone
    
    def __iadd__(self, other):
        
        if not isinstance(other, HistStack):
            raise TypeError("Addition not supported for HistStack and %s"% other.__class__.__name__)
        for hist in other:
            self.Add(hist)
        return self

    def __len__(self):

        return len(self.GetHists())
    
    def __getitem__(self, index):

        return self.GetHists()[index]

    def __iter__(self):

        return iter(self.GetHists())

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
        clone.norm = self.norm
        for hist in self:
            clone.Add(hist.Clone())
        return clone
   
class Hist1D(_HistBase, ROOT.TH1D):
        
    def __init__(self, nbins, bins, name = None, title = None, **kwargs):
        
        if type(bins) not in [list, tuple]:
            raise TypeError()
        if len(bins) < 2:
            raise ValueError()
        if len(bins) == 2:
            if nbins < 1:
                raise ValueError()
            _Object.__init__(self, name, title, nbins, bins[0], bins[1])
        elif len(bins)-1 != nbins:
            raise ValueError()
        else:
            _Object.__init__(self, name, title, nbins, array('d', bins))
        self.decorate(**kwargs)
    
    def GetMaximum(self, include_error = False):

        if not include_error:
            return ROOT.TH1D.GetMaximum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(i+1, clone.GetBinContent(i+1)+clone.GetBinError(i+1))
        return clone.GetMaximum()
    
    def GetMinimum(self, include_error = False):

        if not include_error:
            return ROOT.TH1D.GetMinimum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(i+1, clone.GetBinContent(i+1)-clone.GetBinError(i+1))
        return clone.GetMinimum()
    
    def toGraph(self):

        graph = ROOT.TGraphAsymmErrors(self.hist.Clone(self.GetName()+"_graph"))
        graph.SetName(self.GetName()+"_graph")
        graph.SetTitle(self.GetTitle())
        graph.__class__ = Graph
        graph.integral = self.Integral()
        return graph

    def _content(self):

        return [self.GetBinContent(i) for i in xrange(1, self.GetNbinsX()+1)]

    def __getitem__(self, index):

        _HistBase.__getitem__(self, index)
        return self.GetBinContent(index+1)
    
    def __setitem__(self, index, value):

        _HistBase.__setitem__(self, index)
        self.SetBinContent(index+1, value)

class Hist2D(_HistBase, ROOT.TH2D):

    def __init__(self, nbinsX, binsX, nbinsY, binsY, name = None, title = None, **kwargs):
        
        if type(binsX) not in [list, tuple] or type(binsY) not in [list, tuple]:
            raise TypeError()
        if len(binsX) < 2 or len(binsY) < 2:
            raise ValueError()
        if nbinsX < 1 or nbinsY < 1:
            raise ValueError()
        if len(binsX) == 2 and len(binsY) == 2:
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, binsY[0], binsY[1])
        elif len(binsX) == 2:
            if len(binsY)-1 != nbinsY:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, array('d', binsY))
        elif len(binsY) == 2:
            if len(binsX)-1 != nbinsX:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, binsY[0], binsY[1])
        else:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, array('d', binsY))
        self.decorate(**kwargs)
     
    def _content(self):

        return [[self.GetBinContent(i, j) for i in xrange(1, self.GetNbinsX() + 1)] for j in xrange(1, self.GetNbinsY() + 1)]

    def __getitem__(self, index):
        
        _HistBase.__getitem__(self, index)
        a = ObjectProxy([self.GetBinContent(index+1, j) for j in xrange(1, self.GetNbinsY() + 1)])
        a.__setposthook__('__setitem__', self._setitem(index))
        return a
    
    def _setitem(self, i):
        def __setitem(j, value):
            self.SetBinContent(i+1, j+1, value)
        return __setitem

class Hist3D(_HistBase, ROOT.TH3D):

    def __init__(self, nbinsX, binsX, nbinsY, binsY, nbinsZ, binsZ, name = None, title = None, **kwargs):
        
        if type(binsX) not in [list, tuple] or type(binsY) not in [list, tuple] or type(binsZ) not in [list, tuple]:
            raise TypeError()
        if len(binsX) < 2 or len(binsY) < 2 or len(binsZ) < 2:
            raise ValueError()
        if nbinsX < 1 or nbinsY < 1 or nbinsZ < 1:
            raise ValueError()
        if len(binsX) == 2 and len(binsY) == 2 and len(binsZ) == 2:
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, binsY[0], binsY[1], nbinsZ, binsZ[0], binsZ[1])
        elif len(binsX) == 2 and len(binsY) != 2 and len(binsZ) != 2:
            if len(binsY)-1 != nbinsY or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, array('d', binsY), nbinsZ, array('d', binsZ))
        elif len(binsX) != 2 and len(binsY) == 2 and len(binsZ) != 2:
            if len(binsX)-1 != nbinsX or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, binsY[0], binsY[1], nbinsZ, array('d', binsZ))
        elif len(binsX) != 2 and len(binsY) != 2 and len(binsZ) == 2:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, array('d', binsY), nbinsZ, binsZ[0], binsZ[1])
        elif len(binsX) == 2 and len(binsY) == 2 and len(binsZ) != 2:
            if len(binsZ)-1 != nbinsZ:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, binsY[0], binsY[1], nbinsZ, array('d', binsZ))
        elif len(binsX) == 2 and len(binsY) != 2 and len(binsZ) == 2:
            if len(binsY)-1 != nbinsY:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, binsX[0], binsX[1], nbinsY, array('d', binsY), nbinsZ, binsZ[0], binsZ[1])
        elif len(binsX) != 2 and len(binsY) == 2 and len(binsZ) == 2:
            if len(binsX)-1 != nbinsX:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, binsY[0], binsY[1], nbinsZ, binsZ[0], binsZ[1])
        else:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            _Object.__init__(self, name, title, nbinsX, array('d', binsX), nbinsY, array('d', binsY), nbinsZ, array('d', binsZ))
        self.decorate(**kwargs)
    
    def _content(self):

        return [[[self.GetBinContent(i, j, k) for i in xrange(1, self.GetNbinsX() + 1)] for j in xrange(1, self.GetNbinsY() + 1)] for k in xrange(1, self.GetNbinsZ() + 1)]
    
    def __getitem__(self, index):
        
        _HistBase.__getitem__(self, index)
        out = []
        for j in xrange(1, self.GetNbinsY() + 1):
            a = ObjectProxy([self.GetBinContent(index+1, j, k) for k in xrange(1, self.GetNbinsZ() + 1)])
            a.__setposthook__('__setitem__', self._setitem(index, j-1))
            out.append(a)
        return out
    
    def _setitem(self, i, j):
        def __setitem(k, value):
            self.SetBinContent(i+1, j+1, k+1, value)
        return __setitem

class Graph(_Plottable, _Object, ROOT.TGraphAsymmErrors):
    
    def __init__(self, npoints = 0, file = None, name = None, title = None, **kwargs):

        if npoints > 0:
            _Object.__init__(self, name, title, npoints)
        elif type(file) is str:
            gfile = open(file, 'r')
            lines = gfile.readlines()
            gfile.close()
            _Object.__init__(self, name, title, len(lines)+2)
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
        self.decorate(**kwargs)
    
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

    def getX(self):

        X = self.GetX()
        return [X[i] for i in xrange(self.GetN())]

    def getY(self):
        
        Y = self.GetY()
        return [Y[i] for i in xrange(self.GetN())]

    def getEX(self):

        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        return [(EXlow[i], EXhigh[i]) for i in xrange(self.GetN())]
    
    def getEXhigh(self):

        EXhigh = self.GetEXhigh()
        return [EXhigh[i] for i in xrange(self.GetN())]
    
    def getEXlow(self):

        EXlow = self.GetEXlow()
        return [EXlow[i] for i in xrange(self.GetN())]

    def getEY(self):
        
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        return [(EYlow[i], EYhigh[i]) for i in xrange(self.GetN())]
    
    def getEYhigh(self):
        
        EYhigh = self.GetEYhigh()
        return [EYhigh[i] for i in xrange(self.GetN())]
    
    def getEYlow(self):
        
        EYlow = self.GetEYlow()
        return [EYlow[i] for i in xrange(self.GetN())]

    def GetMaximum(self, include_error = False):

        if not include_error:
            return self.yMax()
        summed = map(add, self.getY(), self.getEYhigh())
        return max(summed)

    def GetMinimum(self, include_error = False):

        if not include_error:
            return self.yMin()
        summed = map(sub, self.getY(), self.getEYlow())
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
                cropGraph.SetPointError(i, EXlow[index], EXhigh[index], EYlow[index], EYhigh[index])
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
            revGraph.SetPointError(i, EXlow[index], EXhigh[index], EYlow[index], EYhigh[index])
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
            invGraph.SetPointError(i, EYlow[i], EYhigh[i], EXlow[i], EXhigh[i])
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
            scaleGraph.SetPointError(i, EXlow[i], EXhigh[i], EYlow[i]*value, EYhigh[i]*value)
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
            stretchGraph.SetPointError(i, EXlow[i]*value, EXhigh[i]*value, EYlow[i], EYhigh[i])
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
            shiftGraph.SetPointError(i, EXlow[i], EXhigh[i], EYlow[i], EYhigh[i])
        return shiftGraph
        
    def Integrate(self):
    
        area = 0.
        X = self.GetX()
        Y = self.GetY()
        for i in xrange(self.GetN()-1):
            area += (X[i+1] - X[i])*(Y[i] + Y[i+1])/2.
        return area

    def Integral(self):

        return self.integral
