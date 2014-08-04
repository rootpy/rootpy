# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..base import NamedObject
from ..decorators import snake_case_methods
from .utils import canvases_with

__all__ = [
    'Axis',
]


@snake_case_methods
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

    def SetRangeUser(self, low, high, update=True):
        if high <= low:
            raise ValueError("high must be greater than low")
        super(Axis, self).SetRangeUser(low, high)
        # Notify relevant canvases that they are modified.
        # Note: some might be missed if our parent is encapsulated in some
        #       other class.
        if not update:
            return
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

    def SetLimits(self, low, high, update=True):
        if high <= low:
            raise ValueError("high must be greater than low")
        super(Axis, self).SetLimits(low, high)
        # Notify relevant canvases that they are modified.
        # Note: some might be missed if our parent is encapsulated in some
        #       other class.
        if not update:
            return
        for c in canvases_with(self.GetParent()):
            c.Modified()
            c.Update()

    @property
    def min(self):
        return self.GetXmin()

    @property
    def max(self):
        return self.GetXmax()

    @min.setter
    def min(self, value):
        # no SetXmin() in ROOT
        self.SetLimits(value, self.GetXmax(), update=False)
        self.SetRangeUser(value, self.GetXmax())

    @max.setter
    def max(self, value):
        # no SetXmax() in ROOT
        self.SetLimits(self.GetXmin(), value, update=False)
        self.SetRangeUser(self.GetXmin(), value)

    @property
    def divisions(self):
        return self.GetNdivisions()

    @divisions.setter
    def divisions(self, value):
        self.SetNdivisions(value)
