import ROOT
import uuid

def isbasictype(thing):

    return isinstance(thing, float) or \
           isinstance(thing, int) or \
           isinstance(thing, long)

class Object(object):

    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__\
            (self, name, title, *args, **kwargs)

    def Clone(self, newName = None):

        if newName:
            clone = self.__class__.__bases__[-1].Clone(self, newName)
        else:
            clone = self.__class__.__bases__[-1]\
                .Clone(self, self.GetName()+'_clone')
        clone.__class__ = self.__class__
        if issubclass(self.__class__, Plottable):
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
        self.format = kwargs.get('format', None)
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

        if isinstance(template_object, Plottable):
            self.decorate(**template_object.__decorators())
        else:
            if isinstance(template_object, ROOT.TAttLine):
                self.linecolor = template_object.GetLineColor()
                self.linestyle = template_object.GetLineStyle()
            if isinstance(template_object, ROOT.TAttFill):
                self.fillcolor = template_object.GetFillColor()
                self.fillstyle = template_object.GetFillStyle()
            if isinstance(template_object, ROOT.TAttMarker):
                self.markercolor = template_object.GetMarkerColor()
                self.markerstyle = template_object.GetMarkerStyle()
       
        if isinstance(self, ROOT.TAttFill):
            if self.fillcolor not in ["white", ""] and \
               self.fillstyle not in ["", "hollow"]:
                self.SetFillStyle(self.fillstyle)
            else:
                self.SetFillStyle("solid")
            self.SetFillColor(self.fillcolor)
        if isinstance(self, ROOT.TAttLine):
            self.SetLineStyle(self.linestyle)
            self.SetLineColor(self.linecolor)
        if isinstance(self, ROOT.TAttMarker):
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
                self.__class__.__bases__[-1].Draw(
                    self, " ".join((self.format, )+args))
            else:
                self.__class__.__bases__[-1].Draw(self, " ".join(args))

