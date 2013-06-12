# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT

from .core import convert_color
from ..core import NamedObject
from .. import QROOT, asrootpy
from ..memory.keepalive import keepalive


class _PadBase(NamedObject):

    def cd(self, *args):
        pad = asrootpy(super(_PadBase, self).cd(*args))
        if pad:
            keepalive(self, pad)
        return pad

    @property
    def primitives(self):
        return asrootpy(self.GetListOfPrimitives())


class Pad(_PadBase, QROOT.TPad):

    def __init__(self, xlow, ylow, xup, yup,
                 color=-1,
                 bordersize=-1,
                 bordermode=-2,
                 name=None,
                 title=None):

        color = convert_color(color, 'root')

        super(Pad, self).__init__(xlow, ylow, xup, yup,
                                  color, bordersize, bordermode,
                                  name=name,
                                  title=title)


class Canvas(_PadBase, QROOT.TCanvas):

    def __init__(self,
                 width=None, height=None,
                 x=None, y=None,
                 name=None, title=None,
                 size_includes_decorations=False):

        # The following line will trigger finalSetup and start the graphics
        # thread if not started already
        style = ROOT.gStyle
        if width is None:
            width = style.GetCanvasDefW()
        if height is None:
            height = style.GetCanvasDefH()
        if x is None:
            x = style.GetCanvasDefX()
        if y is None:
            y = style.GetCanvasDefY()

        super(Canvas, self).__init__(x, y, width, height,
                                     name=name, title=title)

        if not size_includes_decorations:
            # Canvas dimensions include the window manager's decorations by
            # default in vanilla ROOT. I think this is a bad default.
            # Since in the most common case I don't care about the window
            # decorations, the default will be to set the dimensions of the
            # paintable area of the canvas.
            if self.IsBatch():
                self.SetCanvasSize(width, height)
            else:
                self.SetWindowSize(width + (width - self.GetWw()),
                                   height + (height - self.GetWh()))
