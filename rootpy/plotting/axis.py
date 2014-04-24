# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..base import NamedObject
from .utils import canvases_with

__all__ = [
    'Axis',
]


class Axis(NamedObject, QROOT.TAxis):

    _ROOT = QROOT.TAxis

    def __init__(self, name=None, title=None):
        super(Axis, self).__init__(name=name, title=title)

    @property
    def range_user(self):
        first, last = self.GetFirst(), self.GetLast()
        return self.GetBinLowEdge(first), self.GetBinUpEdge(last)

    @range_user.setter
    def range_user(self, r):
        low, high = r
        self.SetRangeUser(low, high)

    def SetRangeUser(self, low, high):
        super(Axis, self).SetRangeUser(low, high)
        # Notify relevant canvases that they are modified.
        # Note: some might be missed if our parent is encapsulated in some
        #       other class.
        for c in canvases_with(self.GetParent()):
            c.Modified()
            c.Update()

    @property
    def limits(self):
        return self.GetXmin(), self.GetXmax()

    @limits.setter
    def limits(self, r):
        low, high = r
        self.SetLimits(low, high)

    def SetLimits(self, low, high):
        super(Axis, self).SetLimits(low, high)
        # Notify relevant canvases that they are modified.
        # Note: some might be missed if our parent is encapsulated in some
        #       other class.
        for c in canvases_with(self.GetParent()):
            c.Modified()
            c.Update()
