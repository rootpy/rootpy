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

ANCHORS = (
    'upper left', 'upper right',
    'lower left', 'lower right',
)


# This is another _PadBase hack. See this comment on github
# https://github.com/rootpy/rootpy/pull/342#issuecomment-19864883

class _Positionable(object):

    def __init__(self, *args, **kwargs):
        self.anchor = kwargs.pop('anchor', 'upper left')
        if self.anchor not in ANCHORS:
            raise ValueError(
                "'{0}' is not a valid anchor position. Use one of {1}".format(
                    self.anchor, ', '.join(ANCHORS)))
        super(_Positionable, self).__init__(*args, **kwargs)

    @property
    def x1(self):
        return self.GetX1()

    @property
    def x2(self):
        return self.GetX2()

    @property
    def y1(self):
        return self.GetY1()

    @property
    def y2(self):
        return self.GetY2()

    @x1.setter
    def x1(self, value):
        self.SetX1(value)

    @x2.setter
    def x2(self, value):
        self.SetX2(value)

    @y1.setter
    def y1(self, value):
        self.SetY1(value)

    @y2.setter
    def y2(self, value):
        self.SetY2(value)

    @property
    def x1_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        width = float(pad.width_pixels)
        return int(self.GetX1() * width)

    @property
    def x2_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        width = float(pad.width_pixels)
        return int(self.GetX2() * width)

    @property
    def y1_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        height = float(pad.height_pixels)
        return int(self.GetY1() * height)

    @property
    def y2_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        height = float(pad.height_pixels)
        return int(self.GetY2() * height)

    @x1_pixels.setter
    def x1_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        width = float(pad.width_pixels)
        self.SetX1(value / width)

    @x2_pixels.setter
    def x2_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        width = float(pad.width_pixels)
        self.SetX2(value / width)

    @y1_pixels.setter
    def y1_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        height = float(pad.height_pixels)
        self.SetY1(value / height)

    @y2_pixels.setter
    def y2_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before setting position in pixels")
        height = float(pad.height_pixels)
        self.SetY2(value / height)

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

    @property
    def height(self):
        return abs(self.GetY2() - self.GetY1())

    @property
    def width(self):
        return abs(self.GetX2() - self.GetX1())

    @height.setter
    def height(self, value):
        if 'upper' in self.anchor:
            self.SetY1(self.GetY2() - value)
        else:
            self.SetY2(self.GetY1() + value)

    @width.setter
    def width(self, value):
        if 'left' in self.anchor:
            self.SetX2(self.GetX1() + value)
        else:
            self.SetX1(self.GetX2() - value)

    @property
    def height_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before getting position in pixels")
        return int(self.height * pad.height_pixels)

    @property
    def width_pixels(self):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before getting position in pixels")
        return int(self.width * pad.width_pixels)

    @height_pixels.setter
    def height_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before getting position in pixels")
        if 'upper' in self.anchor:
            self.SetY1(self.GetY2() - value / float(pad.height_pixels))
        else:
            self.SetY2(self.GetY1() + value / float(pad.height_pixels))

    @width_pixels.setter
    def width_pixels(self, value):
        pad = asrootpy(ROOT.gPad.func())
        if not pad:
            raise RuntimeError(
                "create a pad before getting position in pixels")
        if 'left' in self.anchor:
            self.SetX2(self.GetX1() + value / float(pad.width_pixels))
        else:
            self.SetX1(self.GetX2() - value / float(pad.width_pixels))


class Pave(_Positionable, Object, QROOT.TPave):
    _ROOT = QROOT.TPave


class PaveStats(_Positionable, Object, QROOT.TPaveStats):
    _ROOT = QROOT.TPaveStats
