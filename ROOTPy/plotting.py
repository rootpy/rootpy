from operator import add, sub
import ROOT
from array import array
from style import markers, colours, lines, fills
 
class Graph(ROOT.TGraphAsymmErrors):
    
    def __init__(self,numPoints=0,file=None,name="",title="",**args):

        if numPoints > 0:
            ROOT.TGraphAsymmErrors.__init__(self,numPoints)
        elif type(file) is str:
            gfile = open(file,'r')
            lines = gfile.readlines()
            gfile.close()
            ROOT.TGraphAsymmErrors.__init__(self,len(lines)+2)
            pointIndex = 0
            for line in lines:
                try:
                    X,Y = [float(s) for s in line.strip(" //").split()]
                    self.SetPoint(pointIndex,X,Y)
                    pointIndex += 1
                except: pass
            self.Set(pointIndex)
        else:
            raise ValueError()
        self.SetName(name)
        self.SetTitle(title)
        self.decorate(**args)

    def decorate(self,integral=1.,format="",legend="",marker="circle",markercolour="black",fillcolour="white",visible=True,inlegend=True):

        self.format = format
        self.legend = legend
        self.marker = marker
        self.markercolour = markercolour
        
        self.fillcolour = fillcolour
        self.visible = visible
        self.inlegend = inlegend
        self.integral = integral
        self.intMode = False
  
    def decorators(self):
    
        return {
            "integral":self.integral,
            "format":self.format,
            "legend":self.legend,
            "visible":self.visible,
            "inlegend":self.inlegend,
            "marker":self.marker,
            "markercolour":self.markercolour,
            "fillcolour":self.fillcolour
        }
    
    def __repr__(self):

        return self.__str__()

    def __str__(self):
        
        return "Graph(%s)"%(self.GetTitle())

    def __len__(self): return self.GetN()

    def __getitem__(self,index):

        if index not in range(0,self.GetN()):
            raise IndexError("graph point index out of range")
        return (self.GetX()[index],self.GetY()[index])

    def __setitem__(self,index,point):

        if index not in range(0,self.GetN()):
            raise IndexError("graph point index out of range")
        if type(point) not in [list,tuple]:
            raise TypeError("argument must be a tuple or list")
        if len(point) != 2:
            raise ValueError("argument must be of length 2")
        self.SetPoint(index,point[0],point[1])
    
    def Clone(self,newName=""):
        
        if newName != "":
            clone = ROOT.TGraphAsymmErrors.Clone(self, newName)
        else:
            clone = ROOT.TGraphAsymmErrors.Clone(self, self.GetName()+"_clone")
        clone.__class__ = self.__class__
        clone.decorate(**self.decorators())
        return clone
    
    def Draw(self,options=None):
        
        if self.visible:
            self.SetMarkerStyle(markers[self.marker])
            self.SetMarkerColor(colours[self.markercolour])
            self.SetFillColor(colours[self.fillcolour])
            if not options:
                ROOT.TGraphAsymmErrors.Draw(self, self.format)
            elif type(options) is str:
                ROOT.TGraphAsymmErrors.Draw(self, " ".join([self.format,options]))
            elif type(options) in [list,tuple]:
                ROOT.TGraphAsymmErrors.Draw(self, self.format+" "+" ".join(options))
            else:
                raise TypeError()
    
    def setErrorsFromHist(self,hist):

        if hist.GetNbinsX() != self.GetN(): return
        for i in range(hist.GetNbinsX()):
            content = hist.GetBinContent(i+1)
            if content > 0:
                self.SetPointEYhigh(i,content)
                self.SetPointEYlow(i,0.)
            else:
                self.SetPointEYlow(i,-1*content)
                self.SetPointEYhigh(i,0.)

    def getX(self):

        X = self.GetX()
        return [X[i] for i in xrange(self.GetN())]

    def getY(self):
        
        Y = self.GetY()
        return [Y[i] for i in xrange(self.GetN())]

    def getEX(self):

        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        return [(EXlow[i],EXhigh[i]) for i in xrange(self.GetN())]
    
    def getEXhigh(self):

        EXhigh = self.GetEXhigh()
        return [EXhigh[i] for i in xrange(self.GetN())]
    
    def getEXlow(self):

        EXlow = self.GetEXlow()
        return [EXlow[i] for i in xrange(self.GetN())]


    def getEY(self):
        
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        return [(EYlow[i],EYhigh[i]) for i in xrange(self.GetN())]
    
    def getEYhigh(self):
        
        EYhigh = self.GetEYhigh()
        return [EYhigh[i] for i in xrange(self.GetN())]
    
    def getEYlow(self):
        
        EYlow = self.GetEYlow()
        return [EYlow[i] for i in xrange(self.GetN())]

    def GetMaximum(self,includeError=False):

        if not includeError:
            return self.yMax()
        summed = map(add,self.getY(),self.getEYhigh())
        return max(summed)

    def GetMinimum(self,includeError=False):

        if not includeError:
            return self.yMin()
        summed = map(sub,self.getY(),self.getEYlow())
        return min(summed)
    
    def xMin(self):
        
        if len(self.getX()) == 0:
            raise ValueError("Can't get xmin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(),self.GetX())
    
    def xMax(self):

        if len(self.getX()) == 0:
            raise ValueError("Can't get xmax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(),self.GetX())

    def yMin(self):
        
        if len(self.getY()) == 0:
            raise ValueError("Can't get ymin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(),self.GetY())

    def yMax(self):
    
        if len(self.getY()) == 0:
            raise ValueError("Can't get ymax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(),self.GetY())

    def Crop(self,x1,x2,copy=False):

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
                cropGraph.SetPoint(0,x1,copyGraph.Eval(x1))
            elif i == numPoints - 1 and x2 > xmax:
                cropGraph.SetPoint(i,x2,copyGraph.Eval(x2))
            else:
                cropGraph.SetPoint(i,X[index],Y[index])
                cropGraph.SetPointError(i,EXlow[index],EXhigh[index],EYlow[index],EYhigh[index])
                index += 1
        return cropGraph

    def Reverse(self,copy=False):
        
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
            revGraph.SetPoint(i,X[index],Y[index])
            revGraph.SetPointError(i,EXlow[index],EXhigh[index],EYlow[index],EYhigh[index])
        return revGraph
         
    def Invert(self,copy=False):

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
            invGraph.SetPoint(i,Y[i],X[i])
            invGraph.SetPointError(i,EYlow[i],EYhigh[i],EXlow[i],EXhigh[i])
        return invGraph
 
    def Scale(self,value,copy=False):

        xmin,xmax = self.GetXaxis().GetXmin(),self.GetXaxis().GetXmax()
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
            scaleGraph.SetPoint(i,X[i],Y[i]*value)
            scaleGraph.SetPointError(i,EXlow[i],EXhigh[i],EYlow[i]*value,EYhigh[i]*value)
        scaleGraph.GetXaxis().SetLimits(xmin,xmax)
        scaleGraph.GetXaxis().SetRangeUser(xmin,xmax)
        scaleGraph.integral = self.integral * value
        return scaleGraph

    def Stretch(self,value,copy=False):

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
            stretchGraph.SetPoint(i,X[i]*value,Y[i])
            stretchGraph.SetPointError(i,EXlow[i]*value,EXhigh[i]*value,EYlow[i],EYhigh[i])
        return stretchGraph
    
    def Shift(self,value,copy=False):

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
            shiftGraph.SetPoint(i,X[i]+value,Y[i])
            shiftGraph.SetPointError(i,EXlow[i],EXhigh[i],EYlow[i],EYhigh[i])
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

class HistogramBase(object):
    
    def decorate(self,
        axisLabels=[],
        ylabel="",
        format="EP",
        legend="P",
        intMode=False,
        visible=True,
        inlegend=True,
        markerstyle="circle",
        markercolour="black",
        fillcolour="white",
        fillstyle="hollow",
        linecolour="black",
        linestyle=""):

        self.axisLabels = axisLabels
        self.ylabel = ylabel
        self.format = format
        self.legend = legend
        self.intMode = intMode
        self.visible = visible
        self.inlegend = inlegend
        
        if markers.has_key(markerstyle):
            self.markerstyle = markerstyle
        else:
            self.markerstyle = "circle"

        if colours.has_key(markercolour):
            self.markercolour = markercolour
        else:
            self.markercolour = "black"
        
        if fills.has_key(fillstyle):
            self.fillstyle = fillstyle
        else:
            self.fillstyle = "hollow"
        
        if colours.has_key(fillcolour):
            self.fillcolour = fillcolour
        else:
            self.fillcolour = "white"

        if colours.has_key(linecolour):
            self.linecolour = linecolour
        else:
            self.linecolour = "black"

        if lines.has_key(linestyle):
            self.linestyle = linestyle
        else:
            self.linestyle = ""
     
    def decorators(self):
    
        return {
            "axisLabels":self.axisLabels,
            "ylabel":self.ylabel,
            "format":self.format,
            "legend":self.legend,
            "intMode":self.intMode,
            "visible":self.visible,
            "inlegend":self.inlegend,
            "markercolour":self.markercolour,
            "markerstyle":self.markerstyle,
            "fillcolour":self.fillcolour,
            "fillstyle":self.fillstyle,
            "linecolour":self.linecolour,
            "linestyle":self.linestyle
        }
    
    def Draw(self):

        self.SetMarkerStyle(markers[self.markerstyle])
        self.SetMarkerColor(colours[self.markercolour])
        if self.fillcolour not in ["white",""] and self.fillstyle not in ["","hollow"]:
            self.SetFillStyle(fills[self.fillstyle])
        else:
            self.SetFillStyle(fills["solid"])
        self.SetFillColor(colours[self.fillcolour])
        self.SetLineStyle(lines[self.linestyle])
        self.SetLineColor(colours[self.linecolour])

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "%s(%s)"%(self.__class__.__name__,self.GetTitle())
     
    def __add__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other)
        return copy
        
    def __sub__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other,-1.)
        return copy
        
    def __mul__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        if type(other) in [float,int]:
            copy.Scale(other)
            return copy
        copy.Multiply(other)
        return copy
        
    def __div__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        if type(other) in [float,int]:
            if other == 0:
                raise ZeroDivisionError()
            copy.Scale(1./other)
            return copy
        copy.Divide(other)
        return copy
    
    def __len__(self):

        return self.GetNbinsX()

class Histogram1D(HistogramBase,ROOT.TH1D):
        
    def __init__(self,name,title,nbins,bins,**args):
        
        if type(bins) not in [list,tuple]:
            raise TypeError()
        if len(bins) < 2:
            raise ValueError()
        if len(bins) == 2:
            if nbins < 1:
                raise ValueError()
            ROOT.TH1D.__init__(self,name,title,nbins,bins[0],bins[1])
        elif len(bins)-1 != nbins:
            raise ValueError()
        else:
            ROOT.TH1D.__init__(self,name,title,nbins,array('d',bins))
        self.decorate(**args)
    
    def Clone(self,newName=""):

        if newName != "":
            clone = ROOT.TH1D.Clone(self, newName)
        else:
            clone = ROOT.TH1D.Clone(self, self.GetName()+"_clone")
        clone.__class__ = self.__class__
        clone.decorate(**self.decorators())
        return clone
    
    def Draw(self,options=None):
        
        if type(options) not in [list,tuple]:
            raise TypeError()
        if self.visible:
            HistogramBase.Draw(self)
            if options != None:
                ROOT.TH1D.Draw(self,self.format+" ".join(options))
            else:
                ROOT.TH1D.Draw(self,self.format)

    def GetMaximum(self,includeError=False):

        if not includeError:
            return ROOT.TH1D.GetMaximum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(i+1,clone.GetBinContent(i+1)+clone.GetBinError(i+1))
        return clone.GetMaximum()
    
    def GetMinimum(self,includeError=False):

        if not includeError:
            return ROOT.TH1D.GetMinimum(self)
        clone = self.Clone()
        for i in xrange(clone.GetNbinsX()):
            clone.SetBinContent(i+1,clone.GetBinContent(i+1)-clone.GetBinError(i+1))
        return clone.GetMinimum()
    
    def toGraph(self):

        graph = ROOT.TGraphAsymmErrors(self.hist.Clone(self.GetName()+"_graph"))
        graph.SetName(self.GetName()+"_graph")
        graph.SetTitle(self.GetTitle())
        graph.__class__ = Graph
        graph.integral = self.Integral()
        return graph

class Histogram(Histogram1D): pass

class Histogram2D(HistogramBase,ROOT.TH2D):

    def __init__(self,name,title,nbinsX,binsX,nbinsY,binsY,**args):
        
        if type(binsX) not in [list,tuple] or type(binsY) not in [list,tuple]:
            raise TypeError()
        if len(binsX) < 2 or len(binsY) < 2:
            raise ValueError()
        if nbinsX < 1 or nbinsY < 1:
            raise ValueError()
        if len(binsX) == 2 and len(binsY) == 2:
            ROOT.TH2D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,binsY[0],binsY[1])
        elif len(binsX) == 2:
            if len(binsY)-1 != nbinsY:
                raise ValueError()
            ROOT.TH2D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,array('d',binsY))
        elif len(binsY) == 2:
            if len(binsX)-1 != nbinsX:
                raise ValueError()
            ROOT.TH2D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,binsY[0],binsY[1])
        else:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY:
                raise ValueError()
            ROOT.TH2D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,array('d',binsY))
        self.decorate(**args)
    
    
    def Clone(self,newName=""):

        if newName != "":
            clone = ROOT.TH2D.Clone(self, newName)
        else:
            clone = ROOT.TH2D.Clone(self, self.GetName()+"_clone")
        clone.__class__ = self.__class__
        clone.decorate(**self.decorators())
        return clone
    
    def Draw(self,options=None):
        
        if type(options) not in [list,tuple]:
            raise TypeError()
        if self.visible:
            HistogramBase.Draw(self)
            if options != None:
                ROOT.TH2D.Draw(self,self.format+" ".join(options))
            else:
                ROOT.TH2D.Draw(self,self.format)

# TODO
class Histogram3D(HistogramBase,ROOT.TH3D):

    def __init__(self,name,title,nbinsX,binsX,nbinsY,binsY,nbinsZ,binsZ,**args):
        
        if type(binsX) not in [list,tuple] or type(binsY) not in [list,tuple] or type(binsZ) not in [list,tuple]:
            raise TypeError()
        if len(binsX) < 2 or len(binsY) < 2 or len(binsZ) < 2:
            raise ValueError()
        if nbinsX < 1 or nbinsY < 1 or nbinsZ < 1:
            raise ValueError()
        if len(binsX) == 2 and len(binsY) == 2 and len(binsZ) == 2:
            ROOT.TH3D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,binsY[0],binsY[1],nbinsZ,binsZ[0],binsZ[1])
        elif len(binsX) == 2 and len(binsY) != 2 and len(binsZ) != 2:
            if len(binsY)-1 != nbinsY or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,array('d',binsY),nbinsZ,array('d',binsZ))
        elif len(binsX) != 2 and len(binsY) == 2 and len(binsZ) != 2:
            if len(binsX)-1 != nbinsX or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,binsY[0],binsY[1],nbinsZ,array('d',binsZ))
        elif len(binsX) != 2 and len(binsY) != 2 and len(binsZ) == 2:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,array('d',binsY),nbinsZ,binsZ[0],binsZ[1])
        elif len(binsX) == 2 and len(binsY) == 2 and len(binsZ) != 2:
            if len(binsZ)-1 != nbinsZ:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,binsY[0],binsY[1],nbinsZ,array('d',binsZ))
        elif len(binsX) == 2 and len(binsY) != 2 and len(binsZ) == 2:
            if len(binsY)-1 != nbinsY:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,binsX[0],binsX[1],nbinsY,array('d',binsY),nbinsZ,binsZ[0],binsZ[1])
        elif len(binsX) != 2 and len(binsY) == 2 and len(binsZ) == 2:
            if len(binsX)-1 != nbinsX:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,binsY[0],binsY[1],nbinsZ,binsZ[0],binsZ[1])
        else:
            if len(binsX)-1 != nbinsX or len(binsY)-1 != nbinsY or len(binsZ)-1 != nbinsZ:
                raise ValueError()
            ROOT.TH3D.__init__(self,name,title,nbinsX,array('d',binsX),nbinsY,array('d',binsY),nbinsZ,array('d',binsZ))
        self.decorate(**args)
    
    def Clone(self,newName=""):

        if newName != "":
            clone = ROOT.TH3D.Clone(self, newName)
        else:
            clone = ROOT.TH3D.Clone(self, self.GetName()+"_clone")
        clone.__class__ = self.__class__
        clone.decorate(**self.decorators())
        return clone
    
    def Draw(self,options=None):
        
        if type(options) not in [list,tuple]:
            raise TypeError()
        if self.visible:
            HistogramBase.Draw(self)
            if options != None:
                ROOT.TH3D.Draw(self,self.format+" ".join(options))
            else:
                ROOT.TH3D.Draw(self,self.format)
