# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""
from __future__ import absolute_import

import ROOT

from .base import convert_color
from ..base import NamedObject
from ..context import invisible_canvas
from ..decorators import snake_case_methods
from .. import QROOT, asrootpy
from ..memory.keepalive import keepalive

__all__ = [
    'Pad',
    'Canvas',
]


class _PadBase(NamedObject):

    def cd(self, *args):
        pad = asrootpy(super(_PadBase, self).cd(*args))
        if pad and pad is not self:
            keepalive(self, pad)
        return pad

    @property
    def primitives(self):
        return asrootpy(self.GetListOfPrimitives())

    def __enter__(self):
        self._prev_pad = ROOT.gPad.func()
        self.cd()
        return self

    def __exit__(self, type, value, traceback):
        # similar to preserve_current_canvas in rootpy/context.py
        if self._prev_pad:
            self._prev_pad.cd()
        elif ROOT.gPad.func():
            # Put things back how they were before.
            with invisible_canvas():
                # This is a round-about way of resetting gPad to None.
                # No other technique I tried could do it.
                pass
        self._prev_pad = None
        return False


@snake_case_methods
class Pad(_PadBase, QROOT.TPad):

    _ROOT = QROOT.TPad

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

    def Draw(self, *args):
        ret = super(Pad, self).Draw(*args)
        canvas = self.GetCanvas()
        keepalive(canvas, self)
        return ret


@snake_case_methods
class Canvas(_PadBase, QROOT.TCanvas):

    _ROOT = QROOT.TCanvas

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
