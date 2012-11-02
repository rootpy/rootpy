"""
This module contains base classes defining core funcionality
"""
import warnings
import ROOT
from .style import Color, LineStyle, FillStyle, MarkerStyle
from .canvas import Canvas
from .. import rootpy_globals as _globals


def dim(hist):

    if hasattr(hist, "__dim__"):
        return hist.__dim__()
    return hist.__class__.DIM


class Plottable(object):
    """
    This is a mixin to provide additional attributes for plottable classes
    and to override ROOT TAttXXX and Draw methods.
    """
    EXTRA_ATTRS = {
        'norm': None,
        'drawstyle': '',
        'legendstyle': 'P',
        'integermode': False,
        'visible': True,
        'inlegend': True,
        }

    EXTRA_ATTRS_DEPRECATED = {
        'format': 'drawstyle',
        'intmode': 'integermode',
        }

    @classmethod
    def _get_attr_depr(cls, depattr, newattr):

        def f(self):
            warnings.warn("``%s`` is deprecated and will be removed in "
                        "future versions. Use ``%s`` instead" % (
                            depattr, newattr), DeprecationWarning)
            return getattr(self, newattr)
        return f

    @classmethod
    def _set_attr_depr(cls, depattr, newattr):

        def f(self, value):
            warnings.warn("``%s`` is deprecated and will be removed in "
                        "future versions. Use ``%s`` instead" % (
                            depattr, newattr), DeprecationWarning)
            setattr(self, newattr, value)
        return f

    def __init__(self):

        for attr, value in Plottable.EXTRA_ATTRS.items():
            setattr(self, attr, value)

        for depattr, newattr in Plottable.EXTRA_ATTRS_DEPRECATED.items():
            setattr(Plottable, depattr,
                    property(
                        fget=Plottable._get_attr_depr(depattr, newattr),
                        fset=Plottable._set_attr_depr(depattr, newattr)))

        self.SetMarkerStyle("circle")
        self.SetMarkerColor("black")
        self.SetMarkerSize(1)
        self.SetFillColor("white")
        self.SetFillStyle("hollow")
        self.SetLineColor("black")
        self.SetLineStyle("solid")
        self.SetLineWidth(1)

    def decorate(self, template_object=None, **kwargs):

        if template_object is not None:
            if isinstance(template_object, Plottable):
                self.decorate(**template_object.decorators)
                return
            else:
                if isinstance(template_object, ROOT.TAttLine):
                    self.SetLineColor(template_object.GetLineColor())
                    self.SetLineStyle(template_object.GetLineStyle())
                    self.SetLineWidth(template_object.GetLineWidth())
                if isinstance(template_object, ROOT.TAttFill):
                    self.SetFillColor(template_object.GetFillColor())
                    self.SetFillStyle(template_object.GetFillStyle())
                if isinstance(template_object, ROOT.TAttMarker):
                    self.SetMarkerColor(template_object.GetMarkerColor())
                    self.SetMarkerStyle(template_object.GetMarkerStyle())
                    self.SetMarkerSize(template_object.GetMarkerSize())

        for key, value in kwargs.items():
            if key in Plottable.EXTRA_ATTRS_DEPRECATED:
                newkey = Plottable.EXTRA_ATTRS_DEPRECATED[key]
                warnings.warn("``%s`` is deprecated and will be removed in "
                        "future versions. Use ``%s`` instead" % (
                            key, newkey), DeprecationWarning)
                key = newkey
            if key in Plottable.EXTRA_ATTRS:
                setattr(self, key, value)
            elif key == 'markerstyle':
                self.SetMarkerStyle(value)
            elif key == 'markercolor':
                self.SetMarkerColor(value)
            elif key == 'markersize':
                self.SetMarkerSize(value)
            elif key == 'fillcolor':
                self.SetFillColor(value)
            elif key == 'fillstyle':
                self.SetFillStyle(value)
            elif key == 'linecolor':
                self.SetLineColor(value)
            elif key == 'linestyle':
                self.SetLineStyle(value)
            elif key == 'linewidth':
                self.SetLineWidth(value)
            elif key == 'color':
                self.SetColor(value)
            """
            else:
                raise ValueError("unknown decoration attribute: %s" %
                        key)
            """

    @property
    def decorators(self):

        return {
            "norm": self.norm,
            "drawstyle": self.drawstyle,
            "legendstyle": self.legendstyle,
            "integermode": self.integermode,
            "visible": self.visible,
            "inlegend": self.inlegend,
            "markercolor": self.GetMarkerColor(),
            "markerstyle": self.GetMarkerStyle(),
            "markersize": self.GetMarkerSize(),
            "fillcolor": self.GetFillColor(),
            "fillstyle": self.GetFillStyle(),
            "linecolor": self.GetLineColor(),
            "linestyle": self.GetLineStyle(),
            "linewidth": self.GetLineWidth(),
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

    @property
    def linecolor(self):

        return self.GetLineColor()

    @linecolor.setter
    def linecolor(self, color):

        self.SetLineColor(color)

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

    @property
    def linestyle(self):

        return self.GetLineStyle()

    @linestyle.setter
    def linestyle(self, style):

        self.SetLineStyle(style)

    def SetLineWidth(self, width):

        if isinstance(self, ROOT.TAttLine):
            ROOT.TAttLine.SetLineWidth(self, width)
        else:
            self._linewidth = width

    def GetLineWidth(self):

        if isinstance(self, ROOT.TAttLine):
            return ROOT.TAttLine.GetLineWidth(self)
        else:
            return self._linewidth

    @property
    def linewidth(self):

        return self.GetLineWidth()

    @linewidth.setter
    def linewidth(self, width):

        self.SetLineWidth(width)

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

    @property
    def fillcolor(self):

        return self.GetFillColor()

    @fillcolor.setter
    def fillcolor(self, color):

        self.SetFillColor(color)

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

    @property
    def fillstyle(self):

        return self.GetFillStyle()

    @fillstyle.setter
    def fillstyle(self, style):

        self.SetFillStyle(style)

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

    @property
    def markercolor(self):

        return self.GetMarkerColor()

    @markercolor.setter
    def markercolor(self, color):

        self.SetMarkerColor(color)

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

    @property
    def markerstyle(self):

        return self.GetMarkerStyle()

    @markerstyle.setter
    def markerstyle(self, style):

        self.SetMarkerStyle(style)

    def SetMarkerSize(self, size):

        if isinstance(self, ROOT.TAttMarker):
            ROOT.TAttMarker.SetMarkerSize(self, size)
        else:
            self._markersize = size

    def GetMarkerSize(self):

        if isinstance(self, ROOT.TAttMarker):
            return ROOT.TAttMarker.GetMarkerSize(self)
        else:
            return self._markersize

    @property
    def markersize(self):

        return self.GetMarkerSize()

    @markersize.setter
    def markersize(self, size):

        self.SetMarkerSize(size)

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

    def GetColor(self):

        return self.GetMarkerColor(), self.GetLineColor(), self.GetFillColor()

    @property
    def color(self):

        return self.GetColor()

    @color.setter
    def color(self, color):

        self.SetColor(color)

    def Draw(self, *args):

        if not _globals.pad:
            _globals.pad = Canvas()
        pad = _globals.pad
        pad.cd()
        if hasattr(pad, 'members'):
            if self not in pad.members:
                pad.members.append(self)
        if self.visible:
            if self.drawstyle:
                self.__class__.__bases__[-1].Draw(self,
                        " ".join((self.drawstyle, ) + args))
            else:
                self.__class__.__bases__[-1].Draw(self,
                        " ".join(args))
        pad.Modified()
        pad.Update()
