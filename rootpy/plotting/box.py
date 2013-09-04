# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..core import Object
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
        return (self.GetX1NDC(), self.GetY1NDC(),
                self.GetX2NDC(), self.GetY2NDC())

    @position.setter
    def position(self, value):
        x1, y1, x2, y2 = value
        self.SetX1NDC(x1)
        self.SetY1NDC(y1)
        self.SetX2NDC(x2)
        self.SetY2NDC(y2)

        for c in canvases_with(self):
            c.Modified()


class Pave(_Positionable, Object, QROOT.TPave):
    pass


class PaveStats(_Positionable, Object, QROOT.TPaveStats):
    pass
