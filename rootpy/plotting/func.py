# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..decorators import snake_case_methods
from .base import Plottable
from ..base import NameOnlyObject


__all__ = [
    'F1',
    'F2',
    'F3',
]

class BaseFunction(object):
    class ParProxy(object):
        def __init__(self, fcn, idx):
            self.fcn_ = fcn
            self.idx_ = idx
        
        @property
        def index(self):
            return self.idx_

        @property
        def name(self):
            return self.fcn_.GetParName(self.idx_)
        
        @name.setter
        def name(self, val):
            return self.fcn_.SetParName(self.idx_, val)

        @property
        def value(self):
            return self.fcn_.GetParameter(self.idx_)

        @value.setter
        def value(self, val):
            self.fcn_.SetParameter(self.idx_, val)

        @property
        def error(self):
            return self.fcn_.GetParError(self.idx_)

        @error.setter
        def error(self, val):
            return self.fcn_.SetParError(self.idx_, val)

        @property
        def limits(self):
            m = QROOT.Double()
            M = QROOT.Double()
            self.fcn_.GetParLimits(self.idx_, m, M)
            return float(m), float(M)

        @limits.setter
        def limits(self, val):
            if not hastattr(val, '__len__') and len(val) != 2:
                raise RuntimeError('Function limits must be a tuple size 2')
            self.fcn_.SetParLimits(self.idx_, val[0], val[1])

    def __getitem__(self, value):
        if isinstance(value, basestring):
            idx = self.GetParNumber(value)
        elif isinstance(value, int):
            idx = value
        else:
            raise ValueError('Function index must be a integer or a string')
        return BaseFunction.ParProxy(self, idx)


@snake_case_methods
class F1(Plottable, NameOnlyObject, BaseFunction, QROOT.TF1):
    _ROOT = QROOT.TF1

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        super(F1, self).__init__(*args, name=name)
        self._post_init(**kwargs)


@snake_case_methods
class F2(Plottable, NameOnlyObject, BaseFunction, QROOT.TF2):
    _ROOT = QROOT.TF2

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        super(F2, self).__init__(*args, name=name)
        self._post_init(**kwargs)


@snake_case_methods
class F3(Plottable, NameOnlyObject, BaseFunction, QROOT.TF3):
    _ROOT = QROOT.TF3

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        super(F3, self).__init__(*args, name=name)
        self._post_init(**kwargs)
