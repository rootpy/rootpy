# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import math

from operator import add, sub

import ROOT

from .. import log; log = log[__name__]
from .. import QROOT
from ..core import NamelessConstructorObject, isbasictype
from ..decorators import snake_case_methods
from .core import Plottable


@snake_case_methods
class Graph(Plottable, NamelessConstructorObject, QROOT.TGraphAsymmErrors):

    DIM = 1

    def __init__(self, npoints=0,
                 hist=None,
                 filename=None,
                 name=None,
                 title=None,
                 **kwargs):

        if hist is not None:
            super(Graph, self).__init__(hist, name=name, title=title)
        elif npoints > 0:
            super(Graph, self).__init__(npoints, name=name, title=title)
        elif filename is not None:
            gfile = open(filename, 'r')
            lines = gfile.readlines()
            gfile.close()
            super(Graph, self).__init__(len(lines) + 2, name=name, title=title)
            pointIndex = 0
            for line in lines:
                self.SetPoint(pointIndex,
                              *map(float, line.strip(" //").split()))
                pointIndex += 1
            self.Set(pointIndex)
        else:
            raise ValueError(
                'unable to construct a graph with the supplied arguments')

        self._post_init(**kwargs)

    def __len__(self):

        return self.GetN()

    def __getitem__(self, index):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        return (self.GetX()[index], self.GetY()[index])

    def __setitem__(self, index, point):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        if type(point) not in [list, tuple]:
            raise TypeError("argument must be a tuple or list")
        if len(point) != 2:
            raise ValueError("argument must be of length 2")
        self.SetPoint(index, point[0], point[1])

    def __iter__(self):

        for index in xrange(len(self)):
            yield self[index]

    def x(self, index=None):

        if index is None:
            return (self.GetX()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetX()[index]

    def xerr(self, index=None):

        if index is None:
            return ((self.GetEXlow()[i], self.GetEXhigh()[i])
                    for i in xrange(self.GetN()))
        index = index % len(self)
        return (self.GetErrorXlow(index), self.GetErrorXhigh(index))

    def xerrh(self, index=None):

        if index is None:
            return (self.GetEXhigh()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetErrorXhigh(index)

    def xerrl(self, index=None):

        if index is None:
            return (self.GetEXlow()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetErrorXlow(index)

    def xerravg(self, index=None):

        if index is None:
            return (self.xerravg(i) for i in xrange(self.GetN()))
        index = index % len(self)
        return math.sqrt(self.GetErrorXhigh(index) ** 2 +
                         self.GetErrorXlow(index) ** 2)

    def xedgesl(self, index=None):

        if index is None:
            return (self.xedgesl(i) for i in xrange(self.GetN()))
        index = index % len(self)
        return self.x(index) - self.xerrl(index)

    def xedgesh(self, index=None):

        if index is None:
            return (self.xedgesh(i) for i in xrange(self.GetN()))
        index = index % len(self)
        return self.x(index) + self.xerrh(index)

    def xedges(self):

        for index in xrange(self.GetN()):
            yield self.xedgesl(index)
        yield self.xedgesh(index)

    def y(self, index=None):

        if index is None:
            return (self.GetY()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetY()[index]

    def yerr(self, index=None):

        if index is None:
            return (self.yerr(i) for i in xrange(self.GetN()))
        index = index % len(self)
        return (self.GetErrorYlow(index), self.GetErrorYhigh(index))

    def yerrh(self, index=None):

        if index is None:
            return (self.GetEYhigh()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetEYhigh()[index]

    def yerrl(self, index=None):

        if index is None:
            return (self.GetEYlow()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return self.GetEYlow()[index]

    def yerravg(self, index=None):

        if index is None:
            return (self.yerravg()[i] for i in xrange(self.GetN()))
        index = index % len(self)
        return math.sqrt(self.GetEYhigh()[index] ** 2 +
                         self.GetEYlow()[index] ** 2)

    def __add__(self, other):

        copy = self.Clone()
        copy += other
        return copy

    def __radd__(self, other):

        return self + other

    def __iadd__(self, other):

        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] + other)
        else:
            if len(other) != len(self):
                raise ValueError("graphs do not contain the same number of points")
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = math.sqrt((self.GetEYlow()[index]) ** 2 + (other.GetEYlow()[index]) ** 2)
                yhigh = math.sqrt((self.GetEYhigh()[index]) ** 2 + (other.GetEYhigh()[index]) ** 2)
                self.SetPoint(index, mypoint[0], mypoint[1] + otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __sub__(self, other):

        copy = self.Clone()
        copy -= other
        return copy

    def __rsub__(self, other):

        return -1 * (self - other)

    def __isub__(self, other):

        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                self.SetPoint(index, point[0], point[1] - other)
        else:
            if len(other) != len(self):
                raise ValueError("graphs do not contain the same number of points")
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = math.sqrt((self.GetEYlow()[index]) ** 2 + (other.GetEYlow()[index]) ** 2)
                yhigh = math.sqrt((self.GetEYhigh()[index]) ** 2 + (other.GetEYhigh()[index]) ** 2)
                self.SetPoint(index, mypoint[0], mypoint[1] - otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __div__(self, other):

        copy = self.Clone()
        copy /= other
        return copy

    @staticmethod
    def divide(left, right, consistency=True):

        tmp = left.Clone()
        tmp.__idiv__(right, consistency=consistency)
        return tmp

    def __idiv__(self, other, consistency=True):

        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index]
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1] / other)
                self.SetPointError(index, xlow, xhigh, ylow / other, yhigh / other)
        else:
            if len(other) != len(self) and consistency:
                raise ValueError("graphs do not contain the same number of points")
            if not consistency:
                lowerror = Graph(len(other))
                higherror = Graph(len(other))
                for index, (x, (ylow, yhigh)) in enumerate(zip(other.x(), other.errorsy())):
                    lowerror[index] = (x, ylow)
                    higherror[index] = (x, yhigh)
            for index in xrange(len(self)):
                mypoint = self[index]
                if not consistency:
                    otherpoint = (mypoint[0], other.Eval(mypoint[0]))
                    xlow = self.GetEXlow()[index]
                    xhigh = self.GetEXhigh()[index]
                    ylow = (mypoint[1] / otherpoint[1]) * math.sqrt((self.GetEYlow()[index] / mypoint[1]) ** 2 + (lowerror.Eval(mypoint[0]) / otherpoint[1]) ** 2)
                    yhigh = (mypoint[1] / otherpoint[1]) * math.sqrt((self.GetEYhigh()[index] / mypoint[1]) ** 2 + (higherror.Eval(mypoint[0]) / otherpoint[1]) ** 2)
                elif mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                else:
                    otherpoint = other[index]
                    #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                    #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                    xlow = self.GetEXlow()[index]
                    xhigh = self.GetEXhigh()[index]
                    ylow = (mypoint[1] / otherpoint[1]) * math.sqrt((self.GetEYlow()[index] / mypoint[1]) ** 2 + (other.GetEYlow()[index] / otherpoint[1]) ** 2)
                    yhigh = (mypoint[1] / otherpoint[1]) * math.sqrt((self.GetEYhigh()[index] / mypoint[1]) ** 2 + (other.GetEYhigh()[index] / otherpoint[1]) ** 2)
                self.SetPoint(index, mypoint[0], mypoint[1] / otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def __mul__(self, other):

        copy = self.Clone()
        copy *= other
        return copy

    def __rmul__(self, other):

        return self * other

    def __imul__(self, other):

        if isbasictype(other):
            for index in xrange(len(self)):
                point = self[index]
                ylow, yhigh = self.GetEYlow()[index], self.GetEYhigh()[index]
                xlow, xhigh = self.GetEXlow()[index], self.GetEXhigh()[index]
                self.SetPoint(index, point[0], point[1] * other)
                self.SetPointError(index, xlow, xhigh, ylow * other, yhigh * other)
        else:
            if len(other) != len(self):
                raise ValueError("graphs do not contain the same number of points")
            for index in xrange(len(self)):
                mypoint = self[index]
                otherpoint = other[index]
                if mypoint[0] != otherpoint[0]:
                    raise ValueError("graphs are not compatible: must have same x-coordinate values")
                #xlow = math.sqrt((self.GetEXlow()[index])**2 + (other.GetEXlow()[index])**2)
                #xhigh = math.sqrt((self.GetEXhigh()[index])**2 + (other.GetEXlow()[index])**2)
                xlow = self.GetEXlow()[index]
                xhigh = self.GetEXhigh()[index]
                ylow = (mypoint[1] * otherpoint[1]) * math.sqrt((self.GetEYlow()[index] / mypoint[1]) ** 2 + (other.GetEYlow()[index] / otherpoint[1]) ** 2)
                yhigh = (mypoint[1] * otherpoint[1]) * math.sqrt((self.GetEYhigh()[index] / mypoint[1]) ** 2 + (other.GetEYhigh()[index] / otherpoint[1]) ** 2)
                self.SetPoint(index, mypoint[0], mypoint[1] * otherpoint[1])
                self.SetPointError(index, xlow, xhigh, ylow, yhigh)
        return self

    def setErrorsFromHist(self, hist):

        if hist.GetNbinsX() != self.GetN():
            return
        for i in range(hist.GetNbinsX()):
            content = hist.GetBinContent(i + 1)
            if content > 0:
                self.SetPointEYhigh(i, content)
                self.SetPointEYlow(i, 0.)
            else:
                self.SetPointEYlow(i, -1 * content)
                self.SetPointEYhigh(i, 0.)

    def GetMaximum(self, include_error=False):

        if not include_error:
            return self.yMax()
        summed = map(add, self.y(), self.yerrh())
        return max(summed)

    def maximum(self, include_error=False):

        return self.GetMaximum(include_error)

    def GetMinimum(self, include_error=False):

        if not include_error:
            return self.yMin()
        summed = map(sub, self.y(), self.yerrl())
        return min(summed)

    def minimum(self, include_error=False):

        return self.GetMinimum(include_error)

    def xMin(self):

        if len(self) == 0:
            raise ValueError("Can't get xmin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(), self.GetX())

    def xMax(self):

        if len(self) == 0:
            raise ValueError("Can't get xmax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetX())

    def yMin(self):

        if len(self) == 0:
            raise ValueError("Can't get ymin of empty graph!")
        return ROOT.TMath.MinElement(self.GetN(), self.GetY())

    def yMax(self):

        if len(self) == 0:
            raise ValueError("Can't get ymax of empty graph!")
        return ROOT.TMath.MaxElement(self.GetN(), self.GetY())

    def Crop(self, x1, x2, copy=False):
        """
        Remove points which lie outside of [x1, x2].
        If x1 and/or x2 is below/above the current lowest/highest x-coordinates,
        additional points are added to the graph using a linear interpolation
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
        xmin = copyGraph.xMin()
        if x1 < xmin:
            cropGraph.Set(numPoints + 1)
            numPoints += 1
        xmax = copyGraph.xMax()
        if x2 > xmax:
            cropGraph.Set(numPoints + 1)
            numPoints += 1
        index = 0
        for i in xrange(numPoints):
            if i == 0 and x1 < xmin:
                cropGraph.SetPoint(0, x1, copyGraph.Eval(x1))
            elif i == numPoints - 1 and x2 > xmax:
                cropGraph.SetPoint(i, x2, copyGraph.Eval(x2))
            else:
                cropGraph.SetPoint(i, X[index], Y[index])
                cropGraph.SetPointError(i, EXlow[index], EXhigh[index],
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
        for i in xrange(numPoints):
            index = numPoints - 1 - i
            revGraph.SetPoint(i, X[index], Y[index])
            revGraph.SetPointError(i, EXlow[index], EXhigh[index],
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
        for i in xrange(numPoints):
            invGraph.SetPoint(i, Y[i], X[i])
            invGraph.SetPointError(i, EYlow[i], EYhigh[i],
                                      EXlow[i], EXhigh[i])
        return invGraph

    def Scale(self, value, copy=False):
        """
        Scale the graph vertically by value
        """
        xmin, xmax = self.GetXaxis().GetXmin(), self.GetXaxis().GetXmax()
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
        for i in xrange(numPoints):
            scaleGraph.SetPoint(i, X[i], Y[i] * value)
            scaleGraph.SetPointError(i, EXlow[i], EXhigh[i],
                                        EYlow[i] * value, EYhigh[i] * value)
        scaleGraph.GetXaxis().SetLimits(xmin, xmax)
        scaleGraph.GetXaxis().SetRangeUser(xmin, xmax)
        scaleGraph.integral = self.integral * value
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
        for i in xrange(numPoints):
            stretchGraph.SetPoint(i, X[i] * value, Y[i])
            stretchGraph.SetPointError(i, EXlow[i] * value, EXhigh[i] * value,
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
        for i in xrange(numPoints):
            shiftGraph.SetPoint(i, X[i] + value, Y[i])
            shiftGraph.SetPointError(i, EXlow[i], EXhigh[i],
                                        EYlow[i], EYhigh[i])
        return shiftGraph

    def Integrate(self):
        """
        Integrate using the trapazoidal method
        """
        area = 0.
        X = self.GetX()
        Y = self.GetY()
        for i in xrange(self.GetN() - 1):
            area += (X[i + 1] - X[i]) * (Y[i] + Y[i + 1]) / 2.
        return area


@snake_case_methods
class Graph2D(Plottable, NamelessConstructorObject, QROOT.TGraph2D):

    DIM = 2

    def __init__(self, npoints=0,
                 hist=None,
                 name=None,
                 title=None,
                 **kwargs):

        if hist is not None:
            super(Graph2D, self).__init__(hist, name=name, title=title)
        elif npoints > 0:
            super(Graph2D, self).__init__(npoints, name=name, title=title)
            # ROOT bug in TGraph2D
            self.Set(npoints)
        else:
            raise ValueError(
                'unable to construct a graph with the supplied arguments')
        self._post_init(**kwargs)

    def __len__(self):

        return self.GetN()

    def __getitem__(self, index):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        return (self.GetX()[index], self.GetY()[index], self.GetZ()[index])

    def __setitem__(self, index, point):

        if index not in range(0, self.GetN()):
            raise IndexError("graph point index out of range")
        if type(point) not in [list, tuple]:
            raise TypeError("argument must be a tuple or list")
        if len(point) != 3:
            raise ValueError("argument must be of length 3")
        self.SetPoint(index, point[0], point[1], point[2])

    def __iter__(self):

        for index in xrange(len(self)):
            yield self[index]

    def x(self):

        x = self.GetX()
        for index in xrange(len(self)):
            yield x[index]

    def y(self):

        y = self.GetY()
        for index in xrange(len(self)):
            yield y[index]

    def z(self):

        z = self.GetZ()
        for index in xrange(len(self)):
            yield z[index]
