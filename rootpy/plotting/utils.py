# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

import math
import numpy as np
from .hist import _HistBase
from .graph import Graph


def get_limits(h,
               xpadding=0,
               ypadding=.1,
               xerror_in_padding=True,
               yerror_in_padding=True,
               snap=True,
               logx=False,
               logy=False):

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
        xwidth = math.log(xmax) - math.log(xmin)
        xmin *= 10 ** (- xpadding_bottom * ywidth)
        xmax *= 10 ** (ypadding_top * ywidth)
    else:
        xwidth = xmax - xmin
        xmin -= xpadding_bottom * xwidth
        xmax += xpadding_top * xwidth

    if logy:
        ywidth = math.log(ymax) - math.log(ymin)
        ymin *= 10 ** (- ypadding_bottom * ywidth)
        if snap:
            ymin = min(1, ymin)
        ymax *= 10 ** (ypadding_top * ywidth)
    elif snap and not (ymin < 0 < ymax):
        if ymin >= 0:
            ywidth = ymax
            ymin = 0
            ymax += ypadding_top * ywidth
        elif ymax <= 0:
            ywidth = ymax - ymin
            ymax = 0
            ymin -= ypadding_bottom * ywidth
    else:
        ywidth = ymax - ymin
        ymin -= ypadding_bottom * ywidth
        ymax += ypadding_top * ywidth

    return xmin, xmax, ymin, ymax

