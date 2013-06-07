# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

import ROOT
from math import log
import numpy as np
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
    assert x2 > x1
    assert a + b < 1
    assert a >= 0
    assert b >= 0
    if snap:
        if x1 >= 0:
            x1 = 0
            a = 0
        elif x2 <= 0:
            x2 = 0
            b = 0
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
            "unable to determine plot axes ranges from object of type %s" %
            type(h))

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


def get_band(nom_hist, low_hist, high_hist):
    """
    Convert the low and high histograms into a TGraphAsymmErrors centered at
    the nominal histogram to be used to draw a (possibly asymmetric) error band.
    """
    npoints = len(nom_hist)
    band = Graph(npoints)
    for i in xrange(npoints):
        center = nom_hist.x(i)
        width = nom_hist.xwidth(i)
        nom, low, high = nom_hist[i], low_hist[i], high_hist[i]
        yerrh = max(high - nom, low - nom, 0)
        yerrl = abs(min(high - nom, low - nom, 0))
        band.SetPoint(i, center, nom)
        band.SetPointError(i, width / 2., width / 2.,
                           yerrl, yerrh)
    return band


def canvases_with(drawable):
    """
    Return a list of all canvases where `drawable` has been painted.
    """
    return [c for c in ROOT.gROOT.GetListOfCanvases()
            if drawable in c.GetListOfPrimitives()]
