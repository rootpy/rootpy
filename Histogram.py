import ROOT
from ROOTStyles import markers, colours
from Graph import Graph
from array import array

class Histogram(ROOT.TH1D):
        
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
    
    def decorate(self,axisLabels=[],ylabel="",format="EP",legend="P",intMode=False,visible=True,inlegend=True,marker="circle",colour="black"):

        self.axisLabels = axisLabels
        self.ylabel = ylabel
        self.format = format
        self.legend = legend
        self.intMode = intMode
        self.visible = visible
        self.inlegend = inlegend
        if markers.has_key(marker):
            self.marker = marker
        else:
            self.marker = "circle"
        if colours.has_key(colour):
            self.colour = colour
        else:
            self.colour = "black"
     
    def decorators(self):
    
        return {
            "axisLabels":self.axisLabels,
            "ylabel":self.ylabel,
            "format":self.format,
            "legend":self.legend,
            "intMode":self.intMode,
            "visible":self.visible,
            "inlegend":self.inlegend,
            "marker":self.marker,
            "colour":self.colour
        }

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
            self.SetMarkerStyle(markers[self.marker])
            self.SetMarkerColor(colours[self.colour])
            if options != None:
                ROOT.TH1D.Draw(self,self.format+" ".join(options))
            else:
                ROOT.TH1D.Draw(self,self.format)

    def toGraph(self):

        graph = ROOT.TGraphAsymmErrors(self.hist.Clone(self.GetName()+"_graph"))
        graph.SetName(self.GetName()+"_graph")
        graph.SetTitle(self.GetTitle())
        graph.__class__ = Graph
        graph.integral = self.Integral()
        return graph
    
    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "Histogram(%s)"%(self.GetTitle())
     
    def __add__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other.hist)
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
                raise Exception()
            copy.Scale(1./other)
            return copy
        copy.Divide(other)
        return copy

class Histogram1D(Histogram): pass

class Histogram2D(ROOT.TH2D):

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
    
    def decorate(self,axisLabels=[],ylabel="",format="EP",legend="P",intMode=False,visible=True,inlegend=True,marker="circle",colour="black"):

        self.axisLabels = axisLabels
        self.ylabel = ylabel
        self.format = format
        self.legend = legend
        self.intMode = intMode
        self.visible = visible
        self.inlegend = inlegend
        if markers.has_key(marker):
            self.marker = marker
        else:
            self.marker = "circle"
        if colours.has_key(colour):
            self.colour = colour
        else:
            self.colour = "black"
     
    def decorators(self):
    
        return {
            "axisLabels":self.axisLabels,
            "ylabel":self.ylabel,
            "format":self.format,
            "legend":self.legend,
            "intMode":self.intMode,
            "visible":self.visible,
            "inlegend":self.inlegend,
            "marker":self.marker,
            "colour":self.colour
        }

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
            self.SetMarkerStyle(markers[self.marker])
            self.SetMarkerColor(colours[self.colour])
            if options != None:
                ROOT.TH2D.Draw(self,self.format+" ".join(options))
            else:
                ROOT.TH2D.Draw(self,self.format)
   
    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "Histogram(%s)"%(self.GetTitle())
     
    def __add__(self,other):
        
        copy = self.Clone(self.GetName()+"_clone")
        copy.Add(other.hist)
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
                raise Exception()
            copy.Scale(1./other)
            return copy
        copy.Divide(other)
        return copy
