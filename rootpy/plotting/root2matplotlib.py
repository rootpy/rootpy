from .hist import _HistBase
from .graph import Graph
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'hist',
    'bar',
    'errorbar',
    'fill_between',
]


def _set_defaults(h, kwargs, types=['common']):

    defaults = {}
    for key in types:
        if key == 'common':
            defaults['label'] = h.GetTitle()
            defaults['visible'] = h.visible
        elif key == 'fill':
            defaults['linestyle'] = h.GetLineStyle('mpl')
            defaults['linewidth'] = h.GetLineWidth()
            defaults['edgecolor'] = h.GetLineColor('mpl')
            defaults['facecolor'] = h.GetFillColor('mpl')
            root_fillstyle = h.GetFillStyle('root')
            if root_fillstyle == 0:
                defaults['fill'] = False
            elif root_fillstyle == 1001:
                defaults['fill'] = True
            else:
                defaults['hatch'] = h.GetFillStyle('mpl')
        elif key == 'errors':
            defaults['ecolor'] = h.GetLineColor('mpl')
            defaults['color'] = h.GetMarkerColor('mpl')
            defaults['fmt'] = h.GetMarkerStyle('mpl')
        elif key == 'marker':
            defaults['marker'] = h.GetMarkerStyle('mpl')
            defaults['markersize'] = h.GetMarkerSize() * 5
            defaults['markeredgecolor'] = h.GetMarkerColor('mpl')
            defaults['markerfacecolor'] = h.GetMarkerColor('mpl')
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value


def _set_bounds(h,
                axes=None,
                was_empty=True,
                xpadding=0,
                ypadding=.1,
                snap_zero=True):

    if axes is None:
        axes = plt.gca()

    xmin = h.xedgesl(0)
    xmax = h.xedgesh(-1)
    ymin = min(h)
    ymax = max(h)

    xwidth = xmax - xmin
    if isinstance(xpadding, (tuple, list)):
        if len(xpadding) != 2:
            raise ValueError("xpadding must be of length 2")
        xmin -= xpadding[0] * xwidth
        xmax += xpadding[1] * xwidth
    else:
        xmin -= xpadding * xwidth
        xmax += xpadding * xwidth

    if isinstance(ypadding, (list, tuple)):
        if len(ypadding) != 2:
            raise ValueError("ypadding must be of length 2")
        ypadding_top = ypadding[0]
        ypadding_bottom = ypadding[1]
    else:
        ypadding_top = ypadding_bottom = ypadding

    if snap_zero and not (ymin < 0 < ymax):
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

    if was_empty:
        axes.set_xlim([xmin, xmax])
        axes.set_ylim([ymin, ymax])
    else:
        curr_xmin, curr_xmax = axes.get_xlim()
        axes.set_xlim([min(curr_xmin, xmin), max(curr_xmax, xmax)])
        curr_ymin, curr_ymax = axes.get_ylim()
        axes.set_ylim([min(curr_ymin, ymin), max(curr_ymax, ymax)])


def maybe_reversed(x, reverse=False):

    if reverse:
        return reversed(x)
    return x


def hist(hists, stacked=True, reverse=False, axes=None,
         xpadding=0, ypadding=.1, snap_zero=True,
         **kwargs):
    """
    Make a matplotlib 'step' hist plot.

    *hists* may be a single :class:`rootpy.plotting.hist.Hist` object or a
    :class:`rootpy.plotting.hist.HistStack`.  The *histtype* will be
    set automatically to 'step' or 'stepfilled' for each object based on its
    FillStyle.  All additional keyword arguments will be passed to
    :func:`matplotlib.pyplot.hist`.

    Keyword arguments:

      *stacked*:
        If *True*, the hists will be stacked with the first hist on the bottom.
        If *False*, the hists will be overlaid with the first hist in the
        background.

      *reverse*:
        If *True*, the stacking order will be reversed.
    """
    was_empty = plt.ylim()[1] == 1.
    returns = []
    if isinstance(hists, _HistBase) or isinstance(hists, Graph):
        # This is a single plottable object.
        returns = _hist(hists, axes=axes, **kwargs)
        _set_bounds(hists, axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    elif stacked:
        if axes is None:
            axes = plt.gca()
        for i in range(len(hists)):
            if reverse:
                hsum = sum(hists[i:])
                print hsum.GetFillColor()
            elif i:
                hsum = sum(reversed(hists[:-i]))
            else:
                hsum = sum(reversed(hists))
            # Plot the fill with no edge.
            returns.append(_hist(hsum, **kwargs))
            # Plot the edge with no fill.
            axes.hist(list(hsum.x()), weights=hsum, bins=list(hsum.xedges()),
                      histtype='step', edgecolor=hsum.GetLineColor())
        _set_bounds(sum(hists), axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    else:
        for h in maybe_reversed(hists, reverse):
            returns.append(_hist(h, axes=axes, **kwargs))
        _set_bounds(max(hists), axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    return returns


def _hist(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'fill'])
    kwargs['histtype'] = h.GetFillStyle('root') and 'stepfilled' or 'step'
    return axes.hist(list(h.x()), weights=list(h.y()), bins=list(h.xedges()), **kwargs)


def bar(hists, stacked=True, reverse=False,
        yerr=False, rwidth=0.8, axes=None,
        xpadding=0, ypadding=.1, snap_zero=True,
        **kwargs):
    """
    Make a matplotlib bar plot.

    *hists* may be a single :class:`rootpy.plotting.hist.Hist`, a single
    :class:`rootpy.plotting.graph.Graph`, a list of either type, or a
    :class:`rootpy.plotting.hist.HistStack`.  All additional keyword
    arguments will be passed to :func:`matplotlib.pyplot.bar`.

    Keyword arguments:

      *stacked*:
        If *True*, the hists will be stacked with the first hist on the bottom.
        If *False*, the hists will be overlaid with the first hist in the
        background.  If 'cluster', then the bars will be arranged side-by-side.

      *reverse*:
        If *True*, the stacking order is reversed.

      *yerr*:
        If *False*, no errors are displayed.  If *True*, an individual error will
        be displayed for each hist in the stack.  If 'linear' or 'quadratic', a
        single error bar will be displayed with either the linear or quadratic
        sum of the individual errors.

      *rwidth*:
        The relative width of the bars as a fraction of the bin width.
    """
    was_empty = plt.ylim()[1] == 1.
    returns = []
    nhists = len(hists)
    if isinstance(hists, _HistBase):
        # This is a single histogram.
        returns = _bar(hists, yerr, axes=axes, **kwargs)
        _set_bounds(hists, axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    elif stacked == 'cluster':
        hlist = maybe_reversed(hists, reverse)
        for i, h in enumerate(hlist):
            width = rwidth / nhists
            offset = (1 - rwidth) / 2 + i * width
            returns.append(_bar(h, offset, width, yerr, axes=axes, **kwargs))
        _set_bounds(sum(hists), axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    elif stacked is True:
        hlist = maybe_reversed(hists, reverse)
        bottom, toterr = None, None
        if yerr == 'linear':
            toterr = [sum([h.GetBinError(i + 1) for h in hists])
                      for i in range(len(hists[0]))]
        elif yerr == 'quadratic':
            toterr = [sqrt(sum([h.GetBinError(i + 1) ** 2 for h in hists]))
                      for i in range(len(hists[0]))]
        for i, h in enumerate(hlist):
            err = None
            if yerr is True:
                err = True
            elif yerr and i == (nhists - 1):
                err = toterr
            returns.append(_bar(h, yerr=err, bottom=bottom, axes=axes, **kwargs))
            if bottom:
                bottom += h
            else:
                bottom = h.Clone()
        _set_bounds(bottom, axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    else:
        for h in hlist:
            returns.append(_bar(h, yerr=bool(yerr), axes=axes, **kwargs))
        _set_bounds(max(hists), axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    return returns


def _bar(h, roffset=0., rwidth=1., yerr=None, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    if yerr is True:
        yerr = list(h.yerrors())
    _set_defaults(h, kwargs, ['common', 'fill'])
    width = [x * rwidth for x in h.xwidth()]
    left = [h.xedgesl(i) + h.xwidth(i) * roffset for i in range(len(h))]
    height = h
    return axes.bar(left, height, width, yerr=yerr, **kwargs)


def errorbar(hists, xerr=True, yerr=True, axes=None,
             xpadding=0, ypadding=.1, snap_zero=True,
             emptybins=True,
             **kwargs):
    """
    Make a matplotlib errorbar plot.

    *hists* may be a single :class:`rootpy.plotting.hist.Hist`, a single
    :class:`rootpy.plotting.graph.Graph`, a list of either type, or a
    :class:`rootpy.plotting.hist.HistStack`.  All additional keyword
    arguments will be passed to :func:`matplotlib.pyplot.errorbar`.

    Keyword arguments:

      *xerr/yerr*:
        If *True*, display the x/y errors for each point.
    """
    was_empty = plt.ylim()[1] == 1.
    returns = []
    if isinstance(hists, _HistBase) or isinstance(hists, Graph):
        # This is a single plottable object.
        returns = _errorbar(hists, xerr, yerr,
                axes=axes, emptybins=emptybins, **kwargs)
        _set_bounds(hists, axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    else:
        for h in hists:
            returns.append(_errorbar(h, xerr, yerr,
                axes=axes, emptybins=emptybins, **kwargs))
        _set_bounds(max(hists), axes=axes, was_empty=was_empty,
                    xpadding=xpadding, ypadding=ypadding,
                    snap_zero=snap_zero)
    return returns


def _errorbar(h, xerr, yerr, axes=None, emptybins=True, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'errors', 'marker'])
    if xerr:
        xerr = np.array([list(h.xerrl()), list(h.xerrh())])
    if yerr:
        yerr = np.array([list(h.yerrl()), list(h.yerrh())])
    x = np.array(list(h.x()))
    y = np.array(list(h.y()))
    if not emptybins:
        nonempty = y != 0
        x = x[nonempty]
        y = y[nonempty]
        if xerr is not False:
            xerr = xerr[:, nonempty]
        if yerr is not False:
            yerr = yerr[:, nonempty]
    return axes.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)


def fill_between(high, low, axes=None, **kwargs):
    """
    Fill the region between two histograms or graphs

    *high* and *low* may be a single :class:`rootpy.plotting.hist.Hist`,
    or a single :class:`rootpy.plotting.graph.Graph`. All additional keyword
    arguments will be passed to :func:`matplotlib.pyplot.fill_between`.
    """
    if axes is None:
        axes = plt.gca()
    high_xedges = list(high.xedges())
    low_xedges = list(low.xedges())
    if high_xedges != low_xedges:
        raise ValueError("histogram x edges are incompatible")
    x = []
    top = []
    bottom = []
    for bin in xrange(len(high)):
        x.append(high_xedges[bin])
        top.append(high[bin])
        bottom.append(low[bin])
        x.append(high_xedges[bin + 1])
        top.append(high[bin])
        bottom.append(low[bin])

    return axes.fill_between(x, top, bottom, **kwargs)
