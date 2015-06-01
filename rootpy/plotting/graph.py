# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import math
import numbers
from operator import add, sub

import ROOT

from .. import log; log = log[__name__]
from .. import QROOT
from ..extern.six.moves import range
from ..base import NamelessConstructorObject
from ..decorators import snake_case_methods
from .base import Plottable

__all__ = [
    'Graph',
    'Graph1D',
    'Graph2D',
]


class _GraphBase(object):

    @classmethod
    def from_file(cls, filename, sep=' ', name=None, title=None):
        with open(filename, 'r') as gfile:
            lines = gfile.readlines()
        numpoints = len(lines)
        graph = cls(numpoints, name=name, title=title)
        for idx, line in enumerate(lines):
            point = list(map(float, line.rstrip().split(sep)))
            if len(point) != cls.DIM + 1:
                raise ValueError(
                    "line {0:d} does not contain "
                    "{1:d} values: {2}".format(
                        idx + 1, cls.DIM + 1, line))
            graph.SetPoint(idx, *point)
        graph.Set(numpoints)
        return graph

    def __len__(self):
        return self.GetN()

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def num_points(self):
        return self.GetN()

    @num_points.setter
    def num_points(self, n):
        if n < 0:
            raise ValueError("number of points in a graph must "
                             "be non-negative")
        # ROOT, why not SetN with GetN?
        self.Set(n)

    def x(self, index=None):
        if index is None:
            return (self.GetX()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetX()[index]

    def xerr(self, index=None):
        if index is None:
            return ((self.GetEXlow()[i], self.GetEXhigh()[i])
                    for i in range(self.GetN()))
        index = index % len(self)
        return (self.GetErrorXlow(index), self.GetErrorXhigh(index))

    def xerrh(self, index=None):
        if index is None:
            return (self.GetEXhigh()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetErrorXhigh(index)

    def xerrl(self, index=None):
        if index is None:
            return (self.GetEXlow()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetErrorXlow(index)

    def xerravg(self, index=None):
        if index is None:
            return (self.xerravg(i) for i in range(self.GetN()))
        index = index % len(self)
        return math.sqrt(self.GetErrorXhigh(index) ** 2 +
                         self.GetErrorXlow(index) ** 2)

    def y(self, index=None):
        if index is None:
            return (self.GetY()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetY()[index]

    def yerr(self, index=None):
        if index is None:
            return (self.yerr(i) for i in range(self.GetN()))
        index = index % len(self)
        return (self.GetErrorYlow(index), self.GetErrorYhigh(index))

    def yerrh(self, index=None):
        if index is None:
            return (self.GetEYhigh()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetEYhigh()[index]

    def yerrl(self, index=None):
        if index is None:
            return (self.GetEYlow()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetEYlow()[index]

    def yerravg(self, index=None):
        if index is None:
            return (self.yerravg()[i] for i in range(self.GetN()))
        index = index % len(self)
        return math.sqrt(self.GetEYhigh()[index] ** 2 +
                         self.GetEYlow()[index] ** 2)


class _Graph1DBase(_GraphBase):

    @classmethod
    def divide(cls, top, bottom, option='cp'):
        from .hist import Hist
        if isinstance(top, _Graph1DBase):
            top = Hist(top)
        if isinstance(bottom, _Graph1DBase):
            bottom = Hist(bottom)
        ratio = Graph(type='asymm')
        ratio.Divide(top, bottom, option)
        return ratio

    def __getitem__(self, index):
        if not 0 <= index < self.GetN():
            raise IndexError("graph point index out of range")
        return (self.GetX()[index], self.GetY()[index])

    def __setitem__(self, index, point):
        if not 0 <= index <= self.GetN():
            raise IndexError("graph point index out of range")
        if not isinstance(point, (list, tuple)):
            raise TypeError("argument must be a tuple or list")
        if len(point) != 2:
            raise ValueError("argument must be of length 2")
        self.SetPoint(index, point[0], point[1])

    def __add__(self, other):
        copy = self.Clone()
        copy += other
        return copy

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        copy = self.Clone()
        copy -= other
        return copy

    def __rsub__(self, other):
        return -1 * (self - other)

    def __div__(self, other):
        copy = self.Clone()
        copy /= other
        return copy

    def __mul__(self, other):
        copy = self.Clone()
        copy *= other
        return copy

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        if isinstance(other, numbers.Real):
            for index in range(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] + other)
            return self
        for index in range(len(self)):
            mypoint = self[index]
            otherpoint = other[index]
            xlow = self.GetEXlow()[index]
            xhigh = self.GetEXhigh()[index]
            ylow = math.sqrt((self.GetEYlow()[index]) ** 2 +
                            (other.GetEYlow()[index]) ** 2)
            yhigh = math.sqrt((self.GetEYhigh()[index]) ** 2 +
                                (other.GetEYhigh()[index]) ** 2)
            self.SetPoint(index, mypoint[0], mypoint[1] + otherpoint[1])
            self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __isub__(self, other):
        if isinstance(other, numbers.Real):
            for index in range(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] - other)
            return self
        for index in range(len(self)):
            mypoint = self[index]
            otherpoint = other[index]
            xlow = self.GetEXlow()[index]
            xhigh = self.GetEXhigh()[index]
            ylow = math.sqrt((self.GetEYlow()[index]) ** 2 +
                            (other.GetEYlow()[index]) ** 2)
            yhigh = math.sqrt((self.GetEYhigh()[index]) ** 2 +
                                (other.GetEYhigh()[index]) ** 2)
            self.SetPoint(index, mypoint[0], mypoint[1] - otherpoint[1])
            self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __idiv__(self, other):
        if isinstance(other, numbers.Real):
            for index in range(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index]
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1] / other)
                self.SetPointError(index, xlow, xhigh,
                                   ylow / other, yhigh / other)
            return self
        for index in range(len(self)):
            mypoint = self[index]
            otherpoint = other[index]
            xlow = self.GetEXlow()[index]
            xhigh = self.GetEXhigh()[index]
            ylow = (
                (mypoint[1] / otherpoint[1]) *
                math.sqrt((self.GetEYlow()[index] / mypoint[1]) ** 2 +
                            (other.GetEYlow()[index] /
                                otherpoint[1]) ** 2))
            yhigh = (
                (mypoint[1] / otherpoint[1]) *
                math.sqrt((self.GetEYhigh()[index] / mypoint[1]) ** 2 +
                            (other.GetEYhigh()[index] /
                                otherpoint[1]) ** 2))
            self.SetPoint(index, mypoint[0], mypoint[1] / otherpoint[1])
            self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __imul__(self, other):
        if isinstance(other, numbers.Real):
            for index in range(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index]
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1] * other)
                self.SetPointError(index, xlow, xhigh,
                                   ylow * other, yhigh * other)
            return self
        for index in range(len(self)):
            mypoint = self[index]
            otherpoint = other[index]
            xlow = self.GetEXlow()[index]
            xhigh = self.GetEXhigh()[index]
            ylow = (
                (mypoint[1] * otherpoint[1]) *
                math.sqrt((self.GetEYlow()[index] / mypoint[1]) ** 2 +
                            (other.GetEYlow()[index] / otherpoint[1]) ** 2))
            yhigh = (
                (mypoint[1] * otherpoint[1]) *
                math.sqrt((self.GetEYhigh()[index] / mypoint[1]) ** 2 +
                            (other.GetEYhigh()[index] / otherpoint[1]) ** 2))
            self.SetPoint(index, mypoint[0], mypoint[1] * otherpoint[1])
            self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def GetMaximum(self, include_error=False):
        if not include_error:
            return self.GetYmax()
        summed = map(add, self.y(), self.yerrh())
        return max(summed)

    def GetMinimum(self, include_error=False):
        if not include_error:
            return self.GetYmin()
        summed = map(sub, self.y(), self.yerrl())
        return min(summed)

    def GetXmin(self):
        if len(self) == 0:
            raise ValueError("Attemping to get xmin of empty graph")
        return ROOT.TMath.MinElement(self.GetN(), self.GetX())

    def GetXmax(self):
        if len(self) == 0:
            raise ValueError("Attempting to get xmax of empty graph")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetX())

    def GetYmin(self):
        if len(self) == 0:
            raise ValueError("Attempting to get ymin of empty graph")
        return ROOT.TMath.MinElement(self.GetN(), self.GetY())

    def GetYmax(self):
        if len(self) == 0:
            raise ValueError("Attempting to get ymax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetY())

    def GetEXhigh(self):
        if isinstance(self, ROOT.TGraphErrors):
            return self.GetEX()
        return super(_Graph1DBase, self).GetEXhigh()

    def GetEXlow(self):
        if isinstance(self, ROOT.TGraphErrors):
            return self.GetEX()
        return super(_Graph1DBase, self).GetEXlow()

    def GetEYhigh(self):
        if isinstance(self, ROOT.TGraphErrors):
            return self.GetEY()
        return super(_Graph1DBase, self).GetEYhigh()

    def GetEYlow(self):
        if isinstance(self, ROOT.TGraphErrors):
            return self.GetEY()
        return super(_Graph1DBase, self).GetEYlow()

    def Crop(self, x1, x2, copy=False):
        """
        Remove points which lie outside of [x1, x2].
        If x1 and/or x2 is below/above the current lowest/highest
        x-coordinates, additional points are added to the graph using a
        linear interpolation
        """
        numPoints = self.GetN()
        if copy:
            cropGraph = self.Clone()
            copyGraph = self
        else:
            cropGraph = self
            copyGraph = self.Clone()
        X = copyGraph.GetX()
        EXlow = copyGraph.GetEXlow()
        EXhigh = copyGraph.GetEXhigh()
        Y = copyGraph.GetY()
        EYlow = copyGraph.GetEYlow()
        EYhigh = copyGraph.GetEYhigh()
        xmin = copyGraph.GetXmin()
        if x1 < xmin:
            cropGraph.Set(numPoints + 1)
            numPoints += 1
        xmax = copyGraph.GetXmax()
        if x2 > xmax:
            cropGraph.Set(numPoints + 1)
            numPoints += 1
        index = 0
        for i in range(numPoints):
            if i == 0 and x1 < xmin:
                cropGraph.SetPoint(0, x1, copyGraph.Eval(x1))
            elif i == numPoints - 1 and x2 > xmax:
                cropGraph.SetPoint(i, x2, copyGraph.Eval(x2))
            else:
                cropGraph.SetPoint(i, X[index], Y[index])
                cropGraph.SetPointError(
                    i,
                    EXlow[index], EXhigh[index],
                    EYlow[index], EYhigh[index])
                index += 1
        return cropGraph

    def Reverse(self, copy=False):
        """
        Reverse the order of the points
        """
        numPoints = self.GetN()
        if copy:
            revGraph = self.Clone()
        else:
            revGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in range(numPoints):
            index = numPoints - 1 - i
            revGraph.SetPoint(i, X[index], Y[index])
            revGraph.SetPointError(
                i,
                EXlow[index], EXhigh[index],
                EYlow[index], EYhigh[index])
        return revGraph

    def Invert(self, copy=False):
        """
        Interchange the x and y coordinates of all points
        """
        numPoints = self.GetN()
        if copy:
            invGraph = self.Clone()
        else:
            invGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in range(numPoints):
            invGraph.SetPoint(i, Y[i], X[i])
            invGraph.SetPointError(
                i,
                EYlow[i], EYhigh[i],
                EXlow[i], EXhigh[i])
        return invGraph

    def Scale(self, value, copy=False):
        """
        Scale the graph vertically by value
        """
        numPoints = self.GetN()
        if copy:
            scaleGraph = self.Clone()
        else:
            scaleGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in range(numPoints):
            scaleGraph.SetPoint(i, X[i], Y[i] * value)
            scaleGraph.SetPointError(
                i,
                EXlow[i], EXhigh[i],
                EYlow[i] * value, EYhigh[i] * value)
        return scaleGraph

    def Stretch(self, value, copy=False):
        """
        Stretch the graph horizontally by a factor of value
        """
        numPoints = self.GetN()
        if copy:
            stretchGraph = self.Clone()
        else:
            stretchGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in range(numPoints):
            stretchGraph.SetPoint(i, X[i] * value, Y[i])
            stretchGraph.SetPointError(
                i,
                EXlow[i] * value, EXhigh[i] * value,
                EYlow[i], EYhigh[i])
        return stretchGraph

    def Shift(self, value, copy=False):
        """
        Shift the graph left or right by value
        """
        numPoints = self.GetN()
        if copy:
            shiftGraph = self.Clone()
        else:
            shiftGraph = self
        X = self.GetX()
        EXlow = self.GetEXlow()
        EXhigh = self.GetEXhigh()
        Y = self.GetY()
        EYlow = self.GetEYlow()
        EYhigh = self.GetEYhigh()
        for i in range(numPoints):
            shiftGraph.SetPoint(i, X[i] + value, Y[i])
            shiftGraph.SetPointError(
                i,
                EXlow[i], EXhigh[i],
                EYlow[i], EYhigh[i])
        return shiftGraph

    def Integrate(self):
        """
        Integrate using the trapazoidal method
        """
        area = 0.
        X = self.GetX()
        Y = self.GetY()
        for i in range(self.GetN() - 1):
            area += (X[i + 1] - X[i]) * (Y[i] + Y[i + 1]) / 2.
        return area


class _Graph2DBase(_GraphBase):

    def __getitem__(self, index):
        if not 0 <= index < self.GetN():
            raise IndexError("graph point index out of range")
        return (self.GetX()[index], self.GetY()[index], self.GetZ()[index])

    def __setitem__(self, index, point):
        if not 0 <= index <= self.GetN():
            raise IndexError("graph point index out of range")
        if not isinstance(point, (list, tuple)):
            raise TypeError("argument must be a tuple or list")
        if len(point) != 3:
            raise ValueError("argument must be of length 3")
        self.SetPoint(index, point[0], point[1], point[3])

    def z(self, index=None):
        if index is None:
            return (self.GetZ()[i] for i in range(self.GetN()))
        index = index % len(self)
        return self.GetZ()[index]

    def zerr(self, index=None):
        if index is None:
            return (self.zerr(i) for i in range(self.GetN()))
        index = index % len(self)
        return self.GetErrorZ(index)


_GRAPH1D_BASES = {
    'default': QROOT.TGraph,
    'asymm': QROOT.TGraphAsymmErrors,
    'errors': QROOT.TGraphErrors,
    'benterrors': QROOT.TGraphBentErrors,
}
_GRAPH1D_CLASSES = {}


def _Graph_class(base):

    class Graph(_Graph1DBase, Plottable, NamelessConstructorObject,
                base):
        _ROOT = base
        DIM = 1

        def __init__(self, npoints_or_hist=None,
                     name=None, title=None, **kwargs):
            if npoints_or_hist is not None:
                super(Graph, self).__init__(npoints_or_hist,
                                            name=name, title=title)
            else:
                super(Graph, self).__init__(name=name, title=title)
            self._post_init(**kwargs)

    return Graph

for name, base in _GRAPH1D_BASES.items():
    _GRAPH1D_CLASSES[name] = snake_case_methods(_Graph_class(base))


class Graph(_Graph1DBase, QROOT.TGraph):
    """
    Returns a Graph object which inherits from the associated
    ROOT.TGraph* class (TGraph, TGraphErrors, TGraphAsymmErrors)
    """
    _ROOT = QROOT.TGraph
    DIM = 1

    @classmethod
    def dynamic_cls(cls, type='asymm'):
        return _GRAPH1D_CLASSES[type]

    def __new__(cls, *args, **kwargs):
        type = kwargs.pop('type', 'asymm').lower()
        return cls.dynamic_cls(type)(
            *args, **kwargs)


# alias Graph1D -> Graph
Graph1D = Graph

_GRAPH2D_BASES = {
    'default': QROOT.TGraph2D,
    'errors': QROOT.TGraph2DErrors,
}
_GRAPH2D_CLASSES = {}


def _Graph2D_class(base):

    class Graph2D(_Graph2DBase, Plottable, NamelessConstructorObject,
                base):
        _ROOT = base
        DIM = 2

        def __init__(self, npoints_or_hist=None,
                     name=None, title=None, **kwargs):
            if npoints_or_hist is not None:
                super(Graph2D, self).__init__(npoints_or_hist,
                                              name=name, title=title)
            else:
                super(Graph2D, self).__init__(name=name, title=title)
            if isinstance(npoints_or_hist, int):
                # ROOT bug in TGraph2D
                self.Set(npoints_or_hist)
            self._post_init(**kwargs)

    return Graph2D

for name, base in _GRAPH2D_BASES.items():
    _GRAPH2D_CLASSES[name] = snake_case_methods(_Graph2D_class(base))


class Graph2D(_Graph2DBase, QROOT.TGraph2D):
    """
    Returns a Graph2D object which inherits from the associated
    ROOT.TGraph2D* class (TGraph2D, TGraph2DErrors)
    """
    _ROOT = QROOT.TGraph2D
    DIM = 2

    @classmethod
    def dynamic_cls(cls, type='errors'):
        return _GRAPH2D_CLASSES[type]

    def __new__(cls, *args, **kwargs):
        type = kwargs.pop('type', 'errors').lower()
        return cls.dynamic_cls(type)(
            *args, **kwargs)
