"""
This module contains base classes defining core funcionality
"""

import ROOT
from .style import convert_color, markerstyle, linestyle, fillstyle

def dim(hist):

    if hasattr(hist, "__dim__"):
        return hist.__dim__()
    return hist.__class__.DIM

class Plottable(object):
    """
    This is a mixin to provide additional attributes for plottable classes
    and to override ROOT TAttXXX and Draw methods.
    """

    def __init__(self):

        self.norm  =  None
        self.format = ''
        self.legendstyle = "P"
        self.intmode = False
        self.visible = True
        self.inlegend = True
        
        self.SetMarkerStyle("circle")
        self.SetMarkerColor("black")
        self.SetFillColor("white")
        self.SetFillStyle("hollow")
        self.SetLineColor("black")
        self.SetLineStyle("solid")

    def decorate(self, template_object = None, **kwargs):
        
        self.norm  = kwargs.get('norm', self.norm)
        self.format = kwargs.get('format', self.format)
        self.legendstyle = kwargs.get('legendstyle', self.legendstyle)
        self.intmode = kwargs.get('intmode', self.intmode)
        self.visible = kwargs.get('visible', self.visible)
        self.inlegend = kwargs.get('inlegend', self.inlegend)
        
        markerstyle = kwargs.get('markerstyle', self.markerstyle)
        markercolor = kwargs.get('markercolor', self.markercolor)
        fillcolor = kwargs.get('fillcolor', self.fillcolor)
        fillstyle = kwargs.get('fillstyle', self.fillstyle)
        linecolor = kwargs.get('linecolor', self.linecolor)
        linestyle = kwargs.get('linestyle', self.linestyle)

        if template_object is not None:
            if isinstance(template_object, Plottable):
                self.decorate(**template_object.decorators())
                return
            else:
                if isinstance(template_object, ROOT.TAttLine):
                    linecolor = template_object.GetLineColor()
                    linestyle = template_object.GetLineStyle()
                if isinstance(template_object, ROOT.TAttFill):
                    fillcolor = template_object.GetFillColor()
                    fillstyle = template_object.GetFillStyle()
                if isinstance(template_object, ROOT.TAttMarker):
                    markercolor = template_object.GetMarkerColor()
                    markerstyle = template_object.GetMarkerStyle()
        
        if fillcolor not in ["white", ""] and \
           fillstyle not in ["", "hollow"]:
            self.SetFillStyle(fillstyle)
        else:
            self.SetFillStyle("solid")
        self.SetFillColor(fillcolor)
        self.SetLineStyle(linestyle)
        self.SetLineColor(linecolor)
        self.SetMarkerStyle(markerstyle)
        self.SetMarkerColor(markercolor)
     
    def decorators(self):
    
        return {
            "norm"          : self.norm,
            "format"        : self.format,
            "legendstyle"   : self.legendstyle,
            "intmode"       : self.intmode,
            "visible"       : self.visible,
            "inlegend"      : self.inlegend,
            "markercolor"   : self.GetMarkerColor(),
            "markerstyle"   : self.GetMarkerStyle(),
            "fillcolor"     : self.GetFillColor(),
            "fillstyle"     : self.GetFillStyle(),
            "linecolor"     : self.GetLineColor(),
            "linestyle"     : self.GetLineStyle()
        }

    def SetLineColor(self, color):
        
        if isinstance(self, ROOT.TAttLine):
            self.linecolor = convert_color(color, 'mpl')
            ROOT.TAttLine.SetLineColor(self, convert_color(color, 'root'))

    def GetLineColor(self):

        return self.linecolor
    
    def SetLineStyle(self, style):
        
        if isinstance(self, ROOT.TAttLine):
            self.linestyle = linestyle(style, 'mpl')
            ROOT.TAttLine.SetLineStyle(self, linestyle(style, 'root'))

    def GetLineStyle(self):

        return self.linestyle

    def SetFillColor(self, color):
        
        if isinstance(self, ROOT.TAttFill):
            self.fillcolor = convert_color(color, 'mpl')
            ROOT.TAttFill.SetFillColor(self, convert_color(color, 'root'))

    def GetFillColor(self):

        return self.fillcolor

    def SetFillStyle(self, style):
        
        if isinstance(self, ROOT.TAttFill):
            self.fillstyle = fillstyle(style, 'mpl')
            ROOT.TAttFill.SetFillStyle(self, fillstyle(style, 'root'))
    
    def GetFillStyle(self):

        return self.fillstyle

    def SetMarkerColor(self, color):
        
        if isinstance(self, ROOT.TAttMarker):
            self.markercolor = convert_color(color, 'mpl')
            ROOT.TAttMarker.SetMarkerColor(self, convert_color(color, 'root'))

    def GetMarkerColor(self):

        return self.markercolor

    def SetMarkerStyle(self, style):
        
        if isinstance(self, ROOT.TAttMarker):
            self.markerstyle = markerstyle(style, 'mpl')
            ROOT.TAttMarker.SetMarkerStyle(self, markerstyle(style, 'root'))

    def GetMarkerStyle(self):

        return self.markerstyle

    def Draw(self, *args):
        
        if self.visible:
            if self.format:
                self.__class__.__bases__[-1].Draw(self, " ".join((self.format, )+args))
            else:
                self.__class__.__bases__[-1].Draw(self, " ".join(args))
            pad = ROOT.gPad.cd()
            if hasattr(pad,"members"):
                if self not in pad.members:
                    pad.members.append(self)
