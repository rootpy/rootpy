# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..core import NamedObject
from .utils import canvases_with


class Axis(NamedObject, QROOT.TAxis):

    def __init__(self, name=None, title=None, **kwargs):

        super(Axis, self).__init__(name=name, title=title)

    @property
    def range_user(self):
        first, last = self.GetFirst(), self.GetLast()
        return self.GetBinLowEdge(first), self.GetBinUpEdge(last)

    @range_user.setter
    def range_user(self, r):
        lo, hi = r
        self.SetRangeUser(lo, hi)

    def SetRangeUser(self, lo, hi):

        super(Axis, self).SetRangeUser(lo, hi)

        # Notify relevant canvases that they are modified.
        # Note: some might be missed if our parent is encapsulated in some
        #       other class.

        for c in canvases_with(self.GetParent()):
            c.Modified()
