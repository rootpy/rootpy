"""
This module contains base classes defining core funcionality
"""
import ROOT
import uuid
from rootpy.style import *

def isbasictype(thing):

    return isinstance(thing, float) or \
           isinstance(thing, int) or \
           isinstance(thing, long)

class Object(object):
    """
    Overrides TObject methods. Name and title for TObject-derived classes are optional
    If no name is specified, a UUID is used to ensure uniqueness.
    """
    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__\
            (self, name, title, *args, **kwargs)

    def Clone(self, name = None):

        if name is not None:
            clone = self.__class__.__bases__[-1].Clone(self, name)
        else:
            clone = self.__class__.__bases__[-1].Clone(self, uuid.uuid4().hex)
        clone.__class__ = self.__class__
        if isinstance(self, Plottable):
            clone.decorate(template_object = self)
        return clone

    def __copy__(self):

        return self.Clone()

    def __deepcopy__(self, memo):

        return self.Clone()

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "%s(%s)"%(self.__class__.__name__, self.GetTitle())

class NamelessConstructorObject(Object):
    """
    Handle special cases like TGraph where the
    ROOT constructor does not take name/title
    """
    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__(self, *args, **kwargs)
        self.SetName(name)
        self.SetTitle(title)

class Plottable(object):
    """
    This is a mixin to provide additional attributes for plottable classes
    and to override ROOT TAttXXX and Draw methods.
    """
    def decorate(self, template_object = None, **kwargs):
        
        self.norm  = kwargs.get('norm', None)
        self.format = kwargs.get('format', '')
        self.legendstyle = kwargs.get('legendstyle', "P")
        self.intMode = kwargs.get('intMode', False)
        self.visible = kwargs.get('visible', True)
        self.inlegend = kwargs.get('inLegend', True)
        
        markerstyle = kwargs.get('markerstyle', "circle")
        markercolor = kwargs.get('markercolor', "black")
        fillcolor = kwargs.get('fillcolor', "white")
        fillstyle = kwargs.get('fillstyle', "hollow")
        linecolor = kwargs.get('linecolor', "black")
        linestyle = kwargs.get('linestyle', "")

        if template_object is not None:
            if isinstance(template_object, Plottable):
                self.decorate(**template_object.__decorators())
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
        
        if isinstance(self, ROOT.TAttFill):
            if fillcolor not in ["white", ""] and \
               fillstyle not in ["", "hollow"]:
                self.SetFillStyle(fillstyle)
            else:
                self.SetFillStyle("solid")
            self.SetFillColor(fillcolor)
        if isinstance(self, ROOT.TAttLine):
            self.SetLineStyle(linestyle)
            self.SetLineColor(linecolor)
        if isinstance(self, ROOT.TAttMarker):
            self.SetMarkerStyle(markerstyle)
            self.SetMarkerColor(markercolor)
     
    def __decorators(self):
    
        return {
            "norm"          : self.norm,
            "format"        : self.format,
            "legendstyle"   : self.legendstyle,
            "intMode"       : self.intMode,
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

        if colors.has_key(color):
            self.__class__.__bases__[-1].SetLineColor(self, colors[color])
            self.linecolor = color
        elif color in colors.values():
            self.__class__.__bases__[-1].SetLineColor(self, color)
            self.linecolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetLineColor(self):

        return self.linecolor
    
    def SetLineStyle(self, style):
        
        if lines.has_key(style):
            self.__class__.__bases__[-1].SetLineStyle(self, lines[style])
            self.linestyle = style
        elif style in lines.values():
            self.__class__.__bases__[-1].SetLineStyle(self, style)
            self.linestyle = style
        else:
            raise ValueError("Line style %s not understood"% style)

    def GetLineStyle(self):

        return self.linestyle

    def SetFillColor(self, color):
        
        if colors.has_key(color):
            self.__class__.__bases__[-1].SetFillColor(self, colors[color])
            self.fillcolor = color
        elif color in colors.values():
            self.__class__.__bases__[-1].SetFillColor(self, color)
            self.fillcolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetFillColor(self):

        return self.fillcolor

    def SetFillStyle(self, style):
        
        if fills.has_key(style):
            self.__class__.__bases__[-1].SetFillStyle(self, fills[style])
            self.fillstyle = style
        elif style in fills.values():
            self.__class__.__bases__[-1].SetFillStyle(self, style)
            self.fillstyle = style
        else:
            raise ValueError("Fill style %s not understood"% style)
    
    def GetFillStyle(self):

        return self.fillstyle

    def SetMarkerColor(self, color):
        
        if colors.has_key(color):
            self.__class__.__bases__[-1].SetMarkerColor(self, colors[color])
            self.markercolor = color
        elif color in colors.values():
            self.__class__.__bases__[-1].SetMarkerColor(self, color)
            self.markercolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetMarkerColor(self):

        return self.markercolor

    def SetMarkerStyle(self, style):
        
        if markers.has_key(style):
            self.__class__.__bases__[-1].SetMarkerStyle(self, markers[style])
            self.markerstyle = style
        elif style in markers.values():
            self.__class__.__bases__[-1].SetMarkerStyle(self, style)
            self.markerstyle = style
        else:
            raise ValueError("Marker style %s not understood"% style)

    def GetMarkerStyle(self):

        return self.markerstyle

    def Draw(self, *args):
                
        if self.visible:
            if self.format:
                self.__class__.__bases__[-1].Draw(
                    self, " ".join((self.format, )+args))
            else:
                self.__class__.__bases__[-1].Draw(self, " ".join(args))
