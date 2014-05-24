# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject

from .value import AbsArg

__all__ = [
    'Simultaneous',
    'AddPdf',
    'ProdPdf',
]


class Simultaneous(NamedObject, AbsArg, QROOT.RooSimultaneous):
    _ROOT = QROOT.RooSimultaneous

    def __iter__(self):
        iterator = self.indexCat().typeIterator()
        category = iterator.Next()
        while category:
            yield asrootpy(category)
            category = iterator.Next()

    def getPdf(self, category):
        if isinstance(category, ROOT.RooCatType):
            category = category.GetName()
        return asrootpy(super(Simultaneous, self).getPdf(category))

    def pdf(self, category):
        return self.getPdf(category)

    def indexCat(self):
        return asrootpy(super(Simultaneous, self).indexCat())

    @property
    def index_category(self):
        return self.indexCat()


class AddPdf(NamedObject, AbsArg, QROOT.RooAddPdf):
    _ROOT = QROOT.RooAddPdf


class ProdPdf(NamedObject, AbsArg, QROOT.RooProdPdf):
    _ROOT = QROOT.RooProdPdf
