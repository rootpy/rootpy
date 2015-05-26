# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject
from ..extern.six import string_types

__all__ = [
    'DataSet',
]


class DataSet(NamedObject, QROOT.RooDataSet):
    _ROOT = QROOT.RooDataSet

    def createHistogram(self, *args, **kwargs):
        if args and isinstance(args[0], string_types):
            return ROOT.RooAbsData.createHistogram(self, *args, **kwargs)
        return super(DataSet, self).createHistogram(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return asrootpy(super(DataSet, self).reduce(*args, **kwargs))
