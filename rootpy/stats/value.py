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


class AbsArg(object):
    """
    Use with classes inheriting from RooAbsArg
    """
    def getComponents(self):
        return asrootpy(super(AbsArg, self).getComponents())

    def components(self):
        return self.getComponents()

    def getDependents(self, *args, **kwargs):
        return asrootpy(super(AbsArg, self).getDependents(*args, **kwargs))

    def dependents(self, *args, **kwargs):
        return self.getDependents(*args, **kwargs)

    def getObservables(self, *args, **kwargs):
        return asrootpy(super(AbsArg, self).getObservables(*args, **kwargs))

    def observables(self, *args, **kwargs):
        return self.getObservables(*args, **kwargs)

    def getParameters(self, *args, **kwargs):
        return asrootpy(super(AbsArg, self).getParameters(*args, **kwargs))

    def parameters(self, *args, **kwargs):
        return self.getParameters(*args, **kwargs)


class _ValueBase(object):

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

    @property
    def constant(self):
        return self.getAttribute('Constant')

    @constant.setter
    def constant(self, value):
        self.setConstant(value)


class RealVar(_ValueBase, NamedObject, QROOT.RooRealVar):
    _ROOT = QROOT.RooRealVar
