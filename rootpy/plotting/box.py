# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from .. import QROOT, asrootpy
from ..base import Object
from .utils import canvases_with

__all__ = [
    'Pave',
    'PaveStats',
]


# This is another _PadBase hack. See this comment on github
# https://github.com/rootpy/rootpy/pull/342#issuecomment-19864883

class _Positionable(object):

    @property
    def position(self):
        return (self.GetX1(), self.GetY1(),
                self.GetX2(), self.GetY2())

    @position.setter
    def position(self, value):
        x1, y1, x2, y2 = value
        self.SetX1(x1)
        self.SetY1(y1)
        self.SetX2(x2)
        self.SetY2(y2)
        for c in canvases_with(self):
            c.Modified()

    @property
    def position_pixels(self):
        x1, y1, x2, y2 = self.position
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        width = pad.width_pixels
        height = pad.height_pixels
        return (int(x1 * width), int(y1 * height),
                int(x2 * width), int(y2 * height))

    @position_pixels.setter
    def position_pixels(self, value):
        x1, y1, x2, y2 = value
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before getting position in pixels")
        width = float(pad.width_pixels)
        height = float(pad.height_pixels)
        self.position = (x1 / width, y1 / height,
                         x2 / width, y2 / height)


class Pave(_Positionable, Object, QROOT.TPave):
    _ROOT = QROOT.TPave


class PaveStats(_Positionable, Object, QROOT.TPaveStats):
    _ROOT = QROOT.TPaveStats
