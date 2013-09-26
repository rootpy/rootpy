# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

# trigger ROOT's finalSetup (GUI thread) before matplotlib's
import ROOT
ROOT.kTRUE

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

from .hist import _Hist
from .graph import _Graph1DBase
from .utils import get_limits


__all__ = [
    'hist',
    'bar',
    'errorbar',
    'fill_between',
    'step',
    'hist2d',
    'imshow',
    'contour',
]


def _set_defaults(h, kwargs, types=['common']):

    defaults = {}
    for key in types:
        if key == 'common':
            defaults['label'] = h.GetTitle()
            defaults['visible'] = h.visible
        elif key == 'line':
            defaults['linestyle'] = h.GetLineStyle('mpl')
            defaults['linewidth'] = h.GetLineWidth()
        elif key == 'fill':
            defaults['edgecolor'] = h.GetLineColor('mpl')
            defaults['facecolor'] = h.GetFillColor('mpl')
            root_fillstyle = h.GetFillStyle('root')
            if root_fillstyle == 0:
                defaults['facecolor'] = 'none'
                defaults['fill'] = False
            elif root_fillstyle == 1001:
                defaults['fill'] = True
            else:
                defaults['hatch'] = h.GetFillStyle('mpl')
                defaults['facecolor'] = 'none'
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

    xmin, xmax, ymin, ymax = get_limits(
        h,
        xpadding=xpadding,
        ypadding=ypadding,
        xerror_in_padding=xerror_in_padding,
        yerror_in_padding=yerror_in_padding,
        snap=snap,
        logx=logx,
        logy=logy)

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


def get_highest_zorder(axes):

    return max([c.get_zorder() for c in axes.get_children()])


def maybe_reversed(x, reverse=False):

    if reverse:
        return reversed(x)
    return x


def hist(hists, stacked=True, reverse=False, axes=None,
         xpadding=0, ypadding=.1, yerror_in_padding=True,
         snap=True, logy=None, **kwargs):
    """
    Make a matplotlib 'step' hist plot.

    *hists* may be a single :class:`rootpy.plotting.hist.Hist` object or a
    :class:`rootpy.plotting.hist.HistStack`. All additional keyword arguments
    are passed to :func:`matplotlib.pyplot.fill_between` for the filled regions
    and :func:`matplotlib.pyplot.step` for the edges.

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
    if logy is None:
        logy = axes.get_yscale() == 'log'
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    returns = []
    if isinstance(hists, (_Hist, _Graph1DBase)):
        # This is a single plottable object.
        returns = _hist(hists, axes=axes, logy=logy, **kwargs)
        _set_bounds(hists, axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    elif stacked:
        # draw the top histogram first so its edges don't cover the histograms
        # beneath it in the stack
        if not reverse:
            hists = list(hists)[::-1]
        for i, h in enumerate(hists):
            kwargs_local = kwargs.copy()
            if i == len(hists) - 1:
                low = h.Clone()
                low.Reset()
            else:
                low = sum(hists[i + 1:])
            high = h + low
            proxy = _hist(high, bottom=low, axes=axes, logy=logy, **kwargs)
            returns.append(proxy)
        if not reverse:
            returns = returns[::-1]
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
            returns.append(_hist(h, axes=axes, logy=logy, **kwargs))
        if reverse:
            returns = returns[::-1]
        _set_bounds(max(hists), axes=axes,
                    was_empty=was_empty,
                    prev_xlim=curr_xlim,
                    prev_ylim=curr_ylim,
                    xpadding=xpadding, ypadding=ypadding,
                    yerror_in_padding=yerror_in_padding,
                    snap=snap,
                    logy=logy)
    return returns


def _hist(h, axes=None, bottom=None, logy=None, zorder=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    if zorder is None:
        zorder = get_highest_zorder(axes) + 1

    _set_defaults(h, kwargs, ['common', 'line', 'fill'])
    kwargs_proxy = kwargs.copy()
    fill = kwargs.pop('fill', False) or 'hatch' in kwargs
    if fill:
        # draw the fill without the edge
        if bottom is None:
            bottom = h.Clone()
            bottom.Reset()
        fill_between(bottom, h, axes=axes, logy=logy, linewidth=0,
                     facecolor=kwargs['facecolor'],
                     edgecolor=kwargs['edgecolor'],
                     hatch=kwargs.get('hatch', None),
                     zorder=zorder)
    # draw the edge
    step(h, axes=axes, logy=logy, label=None, zorder=zorder + 1)
    if h.legendstyle.upper() == 'F':
        proxy = plt.Rectangle((0, 0), 0, 0, **kwargs_proxy)
        axes.add_patch(proxy)
    else:
        proxy = plt.Line2D((0, 0), (0, 0),
                           linestyle=kwargs_proxy['linestyle'],
                           linewidth=kwargs_proxy['linewidth'],
                           color=kwargs_proxy['edgecolor'],
                           label=kwargs_proxy['label'])
        axes.add_line(proxy)
    return proxy


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
        If *False*, no y errors are displayed.  If *True*, an individual y
        error will be displayed for each hist in the stack.  If 'linear' or
        'quadratic', a single error bar will be displayed with either the
        linear or quadratic sum of the individual errors.

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
    if isinstance(hists, (_Hist, _Graph1DBase)):
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
            returns.append(_bar(
                h, offset, width,
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
            returns.append(_bar(
                h,
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
    if isinstance(hists, (_Hist, _Graph1DBase)):
        # This is a single plottable object.
        returns = _errorbar(
            hists, xerr, yerr,
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
            errorbar(
                h, xerr=xerr, yerr=yerr, axes=axes,
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


def step(h, axes=None, logy=None, **kwargs):
    """
    Make a matplotlib step plot.
    """
    if axes is None:
        axes = plt.gca()
    if logy is None:
        logy = axes.get_yscale() == 'log'
    _set_defaults(h, kwargs, ['common', 'line'])
    if 'color' not in kwargs:
        kwargs['color'] = h.GetLineColor('mpl')
    y = np.array(list(h) + [0.])
    if logy:
        np.clip(y, 1E-300, 1E300, out=y)
    return axes.step(list(h.xedges()), y, where='post', **kwargs)


def fill_between(a, b, axes=None, logy=None, **kwargs):
    """
    Fill the region between two histograms or graphs

    *a* and *b* may be a single :class:`rootpy.plotting.hist.Hist`,
    or a single :class:`rootpy.plotting.graph.Graph`. All additional keyword
    arguments will be passed to :func:`matplotlib.pyplot.fill_between`.
    """
    if axes is None:
        axes = plt.gca()
    if logy is None:
        logy = axes.get_yscale() == 'log'
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
    if logy:
        np.clip(top, 1E-300, 1E300, out=top)
        np.clip(bottom, 1E-300, 1E300, out=bottom)
    return axes.fill_between(x, top, bottom, **kwargs)


def hist2d(h, axes=None, **kwargs):
    """
    Draw a 2D matplotlib histogram from a 2D ROOT histogram.

    The keyword arguments in `kwargs` are passed directly to
    matplotlib's `hist2d()` function.
    """
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
    """
    Draw a 2D ROOT histogram as a matplotlib imshow plot.

    The keyword arguments in `kwargs` are passed directly to
    matplotlib's `imshow()` function.
    """
    if axes is None:
        axes = plt.gca()
    z = np.array(h.z()).T
    return axes.imshow(
        z,
        extent=[
            h.xedges(0), h.xedges(-1),
            h.yedges(0), h.yedges(-1)],
        interpolation='nearest',
        aspect='auto',
        origin='lower',
        **kwargs)


def contour(h, axes=None, **kwargs):
    """
    Draw a 2D contour plot from a 2D ROOT histogram

    The keyword arguments in `kwargs` are passed directly to
    matplotlib's `contour()` function.
    """
    if axes is None:
        axes = plt.gca()
    x = np.array(list(h.x()))
    y = np.array(list(h.y()))
    z = np.array(h.z()).T
    return axes.contour(x, y, z, **kwargs)
