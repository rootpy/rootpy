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
            Plottable.__init__(clone)
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
        self.SetLineStyle("")

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
     
    def __decorators(self):
    
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

        if colors.has_key(color):
            if isinstance(self, ROOT.TAttLine):
                ROOT.TAttLine.SetLineColor(self, colors[color])
            self.linecolor = color
        elif color in colors.values():
            if isinstance(self, ROOT.TAttLine):
                ROOT.TAttLine.SetLineColor(self, color)
            self.linecolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetLineColor(self):

        return self.linecolor
    
    def SetLineStyle(self, style):
        
        if lines.has_key(style):
            if isinstance(self, ROOT.TAttLine):
                ROOT.TAttLine.SetLineStyle(self, lines[style])
            self.linestyle = style
        elif style in lines.values():
            if isinstance(self, ROOT.TAttLine):
                ROOT.TAttLine.SetLineStyle(self, style)
            self.linestyle = style
        else:
            raise ValueError("Line style %s not understood"% style)

    def GetLineStyle(self):

        return self.linestyle

    def SetFillColor(self, color):
        
        if colors.has_key(color):
            if isinstance(self, ROOT.TAttFill):
                ROOT.TAttFill.SetFillColor(self, colors[color])
            self.fillcolor = color
        elif color in colors.values():
            if isinstance(self, ROOT.TAttFill):
                ROOT.TAttFill.SetFillColor(self, color)
            self.fillcolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetFillColor(self):

        return self.fillcolor

    def SetFillStyle(self, style):
        
        if fills.has_key(style):
            if isinstance(self, ROOT.TAttFill):
                ROOT.TAttFill.SetFillStyle(self, fills[style])
            self.fillstyle = style
        elif style in fills.values():
            if isinstance(self, ROOT.TAttFill):
                ROOT.TAttFill.SetFillStyle(self, style)
            self.fillstyle = style
        else:
            raise ValueError("Fill style %s not understood"% style)
    
    def GetFillStyle(self):

        return self.fillstyle

    def SetMarkerColor(self, color):
        
        if colors.has_key(color):
            if isinstance(self, ROOT.TAttMarker):
                ROOT.TAttMarker.SetMarkerColor(self, colors[color])
            self.markercolor = color
        elif color in colors.values():
            if isinstance(self, ROOT.TAttMarker):
                ROOT.TAttMarker.SetMarkerColor(self, color)
            self.markercolor = color
        else:
            raise ValueError("Color %s not understood"% color)

    def GetMarkerColor(self):

        return self.markercolor

    def SetMarkerStyle(self, style):
        
        if markers.has_key(style):
            if isinstance(self, ROOT.TAttMarker):
                ROOT.TAttMarker.SetMarkerStyle(self, markers[style])
            self.markerstyle = style
        elif style in markers.values():
            if isinstance(self, ROOT.TAttMarker):
                ROOT.TAttMarker.SetMarkerStyle(self, style)
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
