# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from ..base import NamedObject
from .. import QROOT, asrootpy

__all__ = [
    'RealVar',
]


class _ValueBase(NamedObject):

    @property
    def value(self):
        return self.getVal()

    @value.setter
    def value(self, newvalue):
        self.setVal(newvalue)

    @property
    def error(self):
        if self.hasAsymError():
            return self.getErrorHi(), self.getErrorLo()
        return self.getError()

    @error.setter
    def error(self, value):
        if self.hasAsymError():
            # high, low -> low, high
            self.setAsymError(value[1], value[0])
        else:
            self.setError(value)

    @property
    def max(self):
        return self.getMax()

    @max.setter
    def max(self, value):
        self.setMax(value)

    @property
    def min(self):
        return self.getMin()

    @min.setter
    def min(self, value):
        self.setMin(value)


class RealVar(_ValueBase, QROOT.RooRealVar):

    _ROOT = QROOT.RooRealVar
