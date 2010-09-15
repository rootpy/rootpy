from operator import add, sub
import ROOT
from Styles import markers, colours

def getCopy(graph):
    
    if not graph:
        return None
    color = graph.GetFillColor()
    fill = graph.GetFillStyle()
    c = graph.Clone(graph.GetName()+"_clone")
    c.GetXaxis().SetLimits(graph.GetXaxis().GetXmin(),graph.GetXaxis().GetXmax())
    c.GetXaxis().SetRangeUser(graph.GetXaxis().GetXmin(),graph.GetXaxis().GetXmax())
    c.SetFillColor(color)
    c.SetFillStyle(fill)
    return c
 
class Graph(ROOT.TGraphAsymmErrors):
    
    def __init__(self,numPoints=0,file=None,name="",title="",**args):

        if numPoints > 0:
            ROOT.TGraphAsymmErrors.__init__(self,numPoints)
        elif type(file) is str:
            gfile = open(file,'r')
            lines = gfile.readlines()
            gfile.close()
            ROOT.TGraphAsymmErrors.__init__(self,len(lines))
            pointIndex = 0
            for line in lines:
                try:
                    X,Y = [float(s) for s in line.strip(" //").split()]
                    X /= 1000.
                    self.SetPoint(pointIndex,X,Y)
                    pointIndex += 1
                except: pass
            self.Set(pointIndex)
        else:
            raise ValueError()
        self.SetName(name)
        self.SetTitle(title)
        self.decorate(**args)

    def decorate(self,integral=1.,format="",legend="",marker="circle",colour="black",visible=True,inlegend=True):

        self.format = format
        self.legend = legend
        self.marker = marker
        self.colour = colour
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
            "colour":self.colour
        }

    def Clone(self,newName=""):
        
        if newName != "":
            clone = ROOT.TGraphAsymmErrors.Clone(self, newName)
        else:
            clone = ROOT.TGraphAsymmErrors.Clone(self, self.GetName()+"_clone")
        clone.__class__ = self.__class__
        clone.decorate(**self.decorators())
        return clone
    
    def Draw(self,options):
        
        if self.visible:
            self.SetMarkerStyle(markers[self.marker])
            self.SetMarkerColor(colours[self.colour])
            self.SetFillColor(colours[self.colour])
            if type(options) is str:
                ROOT.TGraphAsymmErrors.Draw(self, " ".join([self.format,options]))
            elif typs(options) in [list,tuple]:
                ROOT.TGraphAsymmErrors.Draw(self, self.format+" ".join(options))
            else:
                raise TypeError()
    
    def __repr__(self):

        return self.__str__()

    def __str__(self):
        
        return "Graph(%s)"%(self.GetTitle())
     
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
        
        return ROOT.TMath.MinElement(self.GetN(),self.GetX())
    
    def xMax(self):

        return ROOT.TMath.MaxElement(self.GetN(),self.GetX())

    def yMin(self):
        
        return ROOT.TMath.MinElement(self.GetN(),self.GetY())

    def yMax(self):
    
        return ROOT.TMath.MaxElement(self.GetN(),self.GetY())

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
