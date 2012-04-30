"""
This module contains base classes defining core funcionality
"""

import ROOT
from .style import Color, LineStyle, FillStyle, MarkerStyle
from .. import _globals


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

        self.norm =  None
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

        if template_object is not None:
            if isinstance(template_object, Plottable):
                self.decorate(**template_object.decorators())
                return
            else:
                if isinstance(template_object, ROOT.TAttLine):
                    self.SetLineColor(template_object.GetLineColor())
                    self.SetLineStyle(template_object.GetLineStyle())
                if isinstance(template_object, ROOT.TAttFill):
                    self.SetFillColor(template_object.GetFillColor())
                    self.SetFillStyle(template_object.GetFillStyle())
                if isinstance(template_object, ROOT.TAttMarker):
                    self.SetMarkerColor(template_object.GetMarkerColor())
                    self.SetMarkerStyle(template_object.GetMarkerStyle())

        for key, value in kwargs.items():
            if key in ['norm', 'format', 'legendstyle',
                       'intmode', 'visible', 'inlegend']:
                setattr(self, key, value)
            elif key == 'markerstyle':
                self.SetMarkerStyle(value)
            elif key == 'markercolor':
                self.SetMarkerColor(value)
            elif key == 'fillcolor':
                self.SetFillColor(value)
            elif key == 'fillstyle':
                self.SetFillStyle(value)
            elif key == 'linecolor':
                self.SetLineColor(value)
            elif key == 'linestyle':
                self.SetLineStyle(value)

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
        """
        *color* may be any color understood by ROOT or matplotlib.

        For full documentation of accepted *color* arguments, see
        :class:`rootpy.plotting.style.Color`.
        """
        self._linecolor = Color(color)
        if isinstance(self, ROOT.TAttLine):
            ROOT.TAttLine.SetLineColor(self, self._linecolor('root'))

    def GetLineColor(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._linecolor(mode)

    def SetLineStyle(self, style):
        """
        *style* may be any line style understood by ROOT or matplotlib.

        For full documentation of accepted *style* arguments, see
        :class:`rootpy.plotting.style.LineStyle`.
        """
        self._linestyle = LineStyle(style)
        if isinstance(self, ROOT.TAttLine):
            ROOT.TAttLine.SetLineStyle(self, self._linestyle('root'))

    def GetLineStyle(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._linestyle(mode)

    def SetFillColor(self, color):
        """
        *color* may be any color understood by ROOT or matplotlib.

        For full documentation of accepted *color* arguments, see
        :class:`rootpy.plotting.style.Color`.
        """
        self._fillcolor = Color(color)
        if isinstance(self, ROOT.TAttFill):
            ROOT.TAttFill.SetFillColor(self, self._fillcolor('root'))

    def GetFillColor(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._fillcolor(mode)

    def SetFillStyle(self, style):
        """
        *style* may be any fill style understood by ROOT or matplotlib.

        For full documentation of accepted *style* arguments, see
        :class:`rootpy.plotting.style.FillStyle`.
        """
        self._fillstyle = FillStyle(style)
        if isinstance(self, ROOT.TAttFill):
            ROOT.TAttFill.SetFillStyle(self, self._fillstyle('root'))

    def GetFillStyle(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._fillstyle(mode)

    def SetMarkerColor(self, color):
        """
        *color* may be any color understood by ROOT or matplotlib.

        For full documentation of accepted *color* arguments, see
        :class:`rootpy.plotting.style.Color`.
        """
        self._markercolor = Color(color)
        if isinstance(self, ROOT.TAttMarker):
            ROOT.TAttMarker.SetMarkerColor(self, self._markercolor('root'))

    def GetMarkerColor(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._markercolor(mode)

    def SetMarkerStyle(self, style):
        """
        *style* may be any marker style understood by ROOT or matplotlib.

        For full documentation of accepted *style* arguments, see
        :class:`rootpy.plotting.style.MarkerStyle`.
        """
        self._markerstyle = MarkerStyle(style)
        if isinstance(self, ROOT.TAttMarker):
            ROOT.TAttMarker.SetMarkerStyle(self, self._markerstyle('root'))

    def GetMarkerStyle(self, mode=None):
        """
        *mode* may be 'root', 'mpl', or None to return the ROOT, matplotlib,
        or input value.
        """
        return self._markerstyle(mode)

    def SetColor(self, color):
        """
        *color* may be any color understood by ROOT or matplotlib.

        Set all color attributes with one method call.

        For full documentation of accepted *color* arguments, see
        :class:`rootpy.plotting.style.Color`.
        """
        self.SetFillColor(color)
        self.SetLineColor(color)
        self.SetMarkerColor(color)

    def Draw(self, *args):

        pad = _globals.pad
        if hasattr(pad, 'members'):
            if self not in pad.members:
                pad.members.append(self)
        if self.visible:
            if self.format:
                self.__class__.__bases__[-1].Draw(self, " ".join((self.format, )+args))
            else:
                self.__class__.__bases__[-1].Draw(self, " ".join(args))
