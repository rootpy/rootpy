# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
# trigger ROOT's finalSetup (GUI thread) before matplotlib's
import ROOT
ROOT.kTRUE
from .hist import _HistBase
from .graph import Graph
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math


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
        elif key == 'line':
            defaults['linestyle'] = h.GetLineStyle('mpl')
            defaults['linewidth'] = h.GetLineWidth() * 0.5
        elif key == 'fill':
            defaults['edgecolor'] = h.GetLineColor('mpl')
            defaults['facecolor'] = h.GetFillColor('mpl')
            root_fillstyle = h.GetFillStyle('root')
            if root_fillstyle == 0:
                defaults['fill'] = False
            elif root_fillstyle == 1001:
                defaults['fill'] = True
            else:
                defaults['hatch'] = h.GetFillStyle('mpl')
        elif key == 'marker':
            defaults['marker'] = h.GetMarkerStyle('mpl')
            defaults['markersize'] = h.GetMarkerSize() * 5
            defaults['markeredgecolor'] = h.GetMarkerColor('mpl')
            defaults['markerfacecolor'] = h.GetMarkerColor('mpl')
        elif key == 'errors':
            defaults['ecolor'] = h.GetLineColor('mpl')
        elif key == 'errorbar':
            defaults['fmt'] = h.GetMarkerStyle('mpl')
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value


def _set_bounds(h,
                axes=None,
                was_empty=True,
                prev_xlim=None,
                prev_ylim=None,
                xpadding=0,
                ypadding=.1,
                xerror_in_padding=True,
                yerror_in_padding=True,
                snap=True,
                logx=None,
                logy=None):

    if axes is None:
        axes = plt.gca()

    if prev_xlim is None:
        prev_xlim = plt.xlim()
    if prev_ylim is None:
        prev_ylim = plt.ylim()

    if logx is None:
        logx = axes.get_xscale() == 'log'
    if logy is None:
        logy = axes.get_yscale() == 'log'

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

    if was_empty:
        axes.set_xlim([xmin, xmax])
        axes.set_ylim([ymin, ymax])
    else:
        prev_xmin, prev_xmax = prev_xlim
        if logx and prev_xmin <= 0:
            axes.set_xlim([xmin, max(prev_xmax, xmax)])
        else:
            axes.set_xlim([min(prev_xmin, xmin), max(prev_xmax, xmax)])

        prev_ymin, prev_ymax = prev_ylim
        if logy and prev_ymin <= 0:
            axes.set_ylim([ymin, max(prev_ymax, ymax)])
        else:
            axes.set_ylim([min(prev_ymin, ymin), max(prev_ymax, ymax)])


def maybe_reversed(x, reverse=False):

    if reverse:
        return reversed(x)
    return x


def hist(hists, stacked=True, reverse=False, axes=None,
         xpadding=0, ypadding=.1, yerror_in_padding=True,
         snap=True, **kwargs):
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
    if axes is None:
        axes = plt.gca()
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    logy = kwargs.pop('log', axes.get_yscale() == 'log')
    kwargs['log'] = logy
    returns = []
    if isinstance(hists, _HistBase) or isinstance(hists, Graph):
        # This is a single plottable object.
        returns = _hist(hists, axes=axes, **kwargs)
        _set_bounds(hists, axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    elif stacked:
        if axes is None:
            axes = plt.gca()
        for i in range(len(hists)):
            if reverse:
                hsum = sum(hists[i:])
            elif i:
                hsum = sum(reversed(hists[:-i]))
            else:
                hsum = sum(reversed(hists))
            # Plot the fill with no edge.
            returns.append(_hist(hsum, **kwargs))
            # Plot the edge with no fill.
            axes.hist(list(hsum.x()), weights=hsum, bins=list(hsum.xedges()),
                      histtype='step', edgecolor=hsum.GetLineColor(),
                      log=log_scale)
        _set_bounds(sum(hists), axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    else:
        for h in maybe_reversed(hists, reverse):
            returns.append(_hist(h, axes=axes, **kwargs))
        _set_bounds(max(hists), axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    return returns


def _hist(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'line', 'fill'])
    kwargs['histtype'] = h.GetFillStyle('root') and 'stepfilled' or 'step'
    return axes.hist(list(h.x()), weights=list(h.y()), bins=list(h.xedges()), **kwargs)


def bar(hists, stacked=True, reverse=False,
        xerr=False, yerr=True,
        rwidth=0.8, axes=None,
        xpadding=0, ypadding=.1,
        yerror_in_padding=True,
        snap=True, **kwargs):
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

      *xerr*:
        If *True*, x error bars will be displayed.

      *yerr*:
        If *False*, no y errors are displayed.  If *True*, an individual y error
        will be displayed for each hist in the stack.  If 'linear' or
        'quadratic', a single error bar will be displayed with either the linear
        or quadratic sum of the individual errors.

      *rwidth*:
        The relative width of the bars as a fraction of the bin width.
    """
    if axes is None:
        axes = plt.gca()
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    logy = kwargs.pop('log', axes.get_yscale() == 'log')
    kwargs['log'] = logy
    returns = []
    if isinstance(hists, _HistBase):
        # This is a single histogram.
        returns = _bar(hists, xerr=xerr, yerr=yerr,
                       axes=axes, **kwargs)
        _set_bounds(hists, axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    elif stacked == 'cluster':
        nhists = len(hists)
        hlist = maybe_reversed(hists, reverse)
        for i, h in enumerate(hlist):
            width = rwidth / nhists
            offset = (1 - rwidth) / 2 + i * width
            returns.append(_bar(h, offset, width,
                xerr=xerr, yerr=yerr, axes=axes, **kwargs))
        _set_bounds(sum(hists), axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    elif stacked is True:
        nhists = len(hists)
        hlist = maybe_reversed(hists, reverse)
        toterr = bottom = None
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
            returns.append(_bar(h,
                xerr=xerr, yerr=err,
                bottom=bottom,
                axes=axes, **kwargs))
            if bottom is None:
                bottom = h.Clone()
            else:
                bottom += h
        _set_bounds(bottom, axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    else:
        for h in hlist:
            returns.append(_bar(h, xerr=xerr, yerr=yerr,
                                axes=axes, **kwargs))
        _set_bounds(max(hists), axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    return returns


def _bar(h, roffset=0., rwidth=1., xerr=None, yerr=None, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    if xerr:
        xerr = np.array([list(h.xerrl()), list(h.xerrh())])
    if yerr:
        yerr = np.array([list(h.yerrl()), list(h.yerrh())])
    _set_defaults(h, kwargs, ['common', 'line', 'fill', 'errors'])
    width = [x * rwidth for x in h.xwidth()]
    left = [h.xedgesl(i) + h.xwidth(i) * roffset for i in range(len(h))]
    height = list(h)
    return axes.bar(left, height, width=width, xerr=xerr, yerr=yerr, **kwargs)


def errorbar(hists, xerr=True, yerr=True, axes=None,
             xpadding=0, ypadding=.1,
             xerror_in_padding=True,
             yerror_in_padding=True,
             snap=True,
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
    if axes is None:
        axes = plt.gca()
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    returns = []
    if isinstance(hists, _HistBase) or isinstance(hists, Graph):
        # This is a single plottable object.
        returns = _errorbar(hists, xerr, yerr,
                axes=axes, emptybins=emptybins, **kwargs)
        _set_bounds(hists, axes=axes,
                    was_empty=was_empty,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    xerror_in_padding=xerror_in_padding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap)
    else:
        for h in hists:
            errorbar(h, xerr=xerr, yerr=yerr, axes=axes,
             xpadding=xpadding, ypadding=ypadding,
             xerror_in_padding=xerror_in_padding,
             yerror_in_padding=yerror_in_padding,
             snap=snap,
             emptybins=emptybins,
             **kwargs)
    return returns


def _errorbar(h, xerr, yerr, axes=None, emptybins=True, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'errors', 'errorbar', 'marker'])
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


def step(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'line'])
    return axes.step(list(h.xedges())[:-1], list(h), where='post', **kwargs)


def fill_between(a, b, axes=None, **kwargs):
    """
    Fill the region between two histograms or graphs

    *a* and *b* may be a single :class:`rootpy.plotting.hist.Hist`,
    or a single :class:`rootpy.plotting.graph.Graph`. All additional keyword
    arguments will be passed to :func:`matplotlib.pyplot.fill_between`.
    """
    if axes is None:
        axes = plt.gca()
    log_scale = axes.get_yscale() == 'log'
    a_xedges = list(a.xedges())
    b_xedges = list(b.xedges())
    if a_xedges != b_xedges:
        raise ValueError("histogram x edges are incompatible")
    x = []
    top = []
    bottom = []
    for ibin in xrange(len(a)):
        up = max(a[ibin], b[ibin])
        dn = min(a[ibin], b[ibin])
        x.append(a_xedges[ibin])
        top.append(up)
        bottom.append(dn)
        x.append(a_xedges[ibin + 1])
        top.append(up)
        bottom.append(dn)
    x = np.array(x)
    top = np.array(top)
    bottom = np.array(bottom)
    if log_scale:
        np.clip(top, 1E-300, 1E300, out=top)
        np.clip(bottom, 1E-300, 1E300, out=bottom)
    return axes.fill_between(x, top, bottom, **kwargs)


def hist2d(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    X, Y = np.meshgrid(list(h.x()), list(h.y()))
    x = X.ravel()
    y = Y.ravel()
    z = np.array(h.z()).T
    return axes.hist2d(x, y, weights=z.ravel(),
                       bins=(list(h.xedges()), list(h.yedges())),
                       **kwargs)


def imshow(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    z = np.array(h.z()).T
    return axes.imshow(z,
        extent=[h.xedges(0), h.xedges(-1),
                h.yedges(0), h.yedges(-1)],
        interpolation='nearest',
        aspect='auto',
        origin='lower',
        **kwargs)
