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
    class Entry(object):
        def __init__(self, idx, dataset):
            self.idx_ = idx
            self.dataset_ = dataset
            
        @property
        def fields(self):
            return asrootpy(self.dataset_.get(self.idx_))
        
        @property
        def weight(self):
            self.dataset_.get(self.idx_) #set current event
            return self.dataset_.weight()

    def __len__(self):
        return self.numEntries()

    def __getitem__(self, idx):
        return DataSet.Entry(idx, self)

    def __iter__(self):
        for idx in range(len(self)):
            yield DataSet.Entry(idx, self)

    def createHistogram(self, *args, **kwargs):
        if args and isinstance(args[0], string_types):
            return ROOT.RooAbsData.createHistogram(self, *args, **kwargs)
        return super(DataSet, self).createHistogram(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return asrootpy(super(DataSet, self).reduce(*args, **kwargs))
