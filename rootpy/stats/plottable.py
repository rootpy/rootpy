# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT, asrootpy
from ..base import NamedObject
from ..plotting import Graph
from ..extern.six.moves import range

__all__ = [
    'Plot',
    'Curve',
    'DataHist',
]


class Plot(NamedObject, QROOT.RooPlot):
    _ROOT = QROOT.RooPlot

    @property
    def objects(self):
        for i in range(int(self.numItems())):
            yield asrootpy(self.getObject(i))

    @property
    def curves(self):
        for obj in self.objects:
            if isinstance(obj, Curve):
                yield obj

    @property
    def data_hists(self):
        for obj in self.objects:
            if isinstance(obj, DataHist):
                yield obj

    @property
    def plotvar(self):
        return asrootpy(self.getPlotVar())


class Curve(NamedObject, Graph, QROOT.RooCurve):
    _ROOT = QROOT.RooCurve


class DataHist(NamedObject, Graph, QROOT.RooHist):
    _ROOT = QROOT.RooHist
