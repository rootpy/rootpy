# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from math import log

import ROOT

from .hist import _HistBase, HistStack
from .graph import Graph


def _limits_helper(x1, x2, a, b, snap=False):
    """
    Given x1, x2, a, b, where:

        x1 - x0         x3 - x2
    a = ------- ,   b = -------
        x3 - x0         x3 - x0

    determine the points x0 and x3:

    x0         x1                x2       x3
    |----------|-----------------|--------|

    """
    if x2 < x1:
        raise ValueError("x2 < x1")
    if a + b >= 1:
        raise ValueError("a + b >= 1")
    if a < 0:
        raise ValueError("a < 0")
    if b < 0:
        raise ValueError("b < 0")
    if snap:
        if x1 >= 0:
            x1 = 0
            a = 0
        elif x2 <= 0:
            x2 = 0
            b = 0
        if x1 == x2 == 0:
            raise ValueError(
                "range is ambiguous when x1 == x2 == 0 and snap=True")
    elif x1 == x2:
        raise ValueError(
            "range is ambiguous when x1 == x2 and snap=False")
    if a == 0 and b == 0:
        return x1, x2
    elif a == 0:
        return x1, (x2 - b * x1) / (1 - b)
    elif b == 0:
        return (x1 - a * x2) / (1 - a), x2
    x0 = ((b / a) * x1 + x2 - (x2 - x1) / (1 - a - b)) / (1 + b / a)
    x3 = (x2 - x1) / (1 - a - b) + x0
    return x0, x3


def get_limits(h,
               xpadding=0,
               ypadding=.1,
               xerror_in_padding=True,
               yerror_in_padding=True,
               snap=True,
               logx=False,
               logy=False):
    """
    Get the axes limits that should be used for a histogram or graph
    """
    import numpy as np

    if isinstance(h, HistStack):
        h = h.sum

    if isinstance(h, (_HistBase, Graph)):
        y_array_min = y_array_max = np.array(list(h.y()))
        if yerror_in_padding:
            y_array_min = y_array_min - np.array(list(h.yerrl()))
            y_array_max = y_array_max + np.array(list(h.yerrh()))
        if logy:
            y_array_min = y_array_min[y_array_min > 0]
        ymin = y_array_min.min()
        ymax = y_array_max.max()
        if isinstance(h, Graph):
            x_array_min = x_array_max = np.array(list(h.x()))
            if xerror_in_padding:
                x_array_min = x_array_min - np.array(list(h.xerrl()))
                x_array_max = x_array_max + np.array(list(h.xerrh()))
            if logx:
                x_array_min = x_array_min[x_array_min > 0]
            xmin = x_array_min.min()
            xmax = x_array_max.max()
        else:
            xmin = h.xedgesl(0)
            xmax = h.xedgesh(-1)
    else:
        raise TypeError(
            "unable to determine plot axes ranges "
            "from object of type `{0}`".format(
                type(h)))

    if isinstance(xpadding, (list, tuple)):
        if len(xpadding) != 2:
            raise ValueError("xpadding must be of length 2")
        xpadding_top = xpadding[0]
        xpadding_bottom = xpadding[1]
    else:
        xpadding_top = xpadding_bottom = xpadding

    if isinstance(ypadding, (list, tuple)):
        if len(ypadding) != 2:
            raise ValueError("ypadding must be of length 2")
        ypadding_top = ypadding[0]
        ypadding_bottom = ypadding[1]
    else:
        ypadding_top = ypadding_bottom = ypadding

    if logx:
        x0, x3 = _limits_helper(log(xmin), log(xmax),
                                xpadding_bottom, xpadding_top)
        xmin = 10 ** x0
        xmax = 10 ** x3
    else:
        xmin, xmax = _limits_helper(xmin, xmax, xpadding_bottom, xpadding_top)

    if logy:
        y0, y3 = _limits_helper(log(ymin), log(ymax),
                                ypadding_bottom, ypadding_top, snap=snap)
        ymin = 10 ** y0
        ymax = 10 ** y3
    else:
        ymin, ymax = _limits_helper(ymin, ymax, ypadding_bottom, ypadding_top,
                                    snap=snap)

    return xmin, xmax, ymin, ymax


def get_band(low_hist, high_hist, middle_hist=None):
    """
    Convert the low and high histograms into a TGraphAsymmErrors centered at
    the middle histogram if not None otherwise the middle between the low and
    high points, to be used to draw a (possibly asymmetric) error band.
    """
    npoints = len(low_hist)
    band = Graph(npoints)
    for i in xrange(npoints):
        center = low_hist.x(i)
        width = low_hist.xwidth(i)
        low, high = low_hist[i], high_hist[i]
        if middle_hist is not None:
            middle = middle_hist[i]
        else:
            middle = (low + high) / 2.
        yerrh = max(high - middle, low - middle, 0)
        yerrl = abs(min(high - middle, low - middle, 0))
        band.SetPoint(i, center, middle)
        band.SetPointError(i, width / 2., width / 2.,
                           yerrl, yerrh)
    return band


def all_primitives(pad):
    """
    Recursively find all primities on a canvas, even those hiding behind a
    GetListOfFunctions() of a primitive
    """
    result = []
    for primitive in pad.GetListOfPrimitives():
        result.append(primitive)
        if hasattr(primitive, "GetListOfFunctions"):
            result.extend(primitive.GetListOfFunctions())
        if hasattr(primitive, "GetHistogram"):
            p = primitive.GetHistogram()
            if p:
                result.append(p)
        if isinstance(primitive, ROOT.TPad):
            result.extend(all_primitives(primitive))
    return result


def canvases_with(drawable):
    """
    Return a list of all canvases where `drawable` has been painted.

    Note: This function is inefficient because it inspects all objects on all
          canvases, recursively. Avoid calling it if you have a large number of
          canvases and primitives.
    """
    return [c for c in ROOT.gROOT.GetListOfCanvases()
            if drawable in all_primitives(c)]
