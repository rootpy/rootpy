# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject
from .value import AbsArg

__all__ = [
    'CatType',
    'Category',
]


class CatType(NamedObject, QROOT.RooCatType):
    _ROOT = QROOT.RooCatType

    @property
    def value(self):
        return self.getVal()

    @value.setter
    def value(self, val):
        self.setVal(val)


class Category(NamedObject, AbsArg, QROOT.RooCategory):
    _ROOT = QROOT.RooCategory

    @property
    def index(self):
        return self.getIndex()

    @index.setter
    def index(self, value):
        return self.setIndex(value)

    @property
    def label(self):
        return self.getLabel()

    @index.setter
    def label(self, value):
        return self.setLabel(value)
