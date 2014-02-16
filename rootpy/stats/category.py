# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject

__all__ = [
    'CatType',
]


class CatType(NamedObject, QROOT.RooCatType):

    _ROOT = QROOT.RooCatType

    @property
    def value(self):
        return self.getVal()

    @value.setter
    def value(self, val):
        self.setVal(val)
