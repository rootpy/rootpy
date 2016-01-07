# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module provides functions that allow the plotting of ROOT histograms and
graphs with `matplotlib <http://matplotlib.org/>`_.

If you just want to save image files and don't want matplotlib to attempt to
create a graphical window, tell matplotlib to use a non-interactive backend
such as ``Agg`` when importing it for the first time (i.e. before importing
rootpy.plotting.root2matplotlib)::

   import matplotlib
   matplotlib.use('Agg') # do this before importing pyplot or root2matplotlib

This puts matplotlib in a batch state similar to ``ROOT.gROOT.SetBatch(True)``.
"""
from __future__ import absolute_import

# trigger ROOT's finalSetup (GUI thread) before matplotlib's
import ROOT
ROOT.kTRUE

from math import sqrt
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import matplotlib.pyplot as plt
import numpy as np

from ..extern.six.moves import range
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


def _set_defaults(obj, kwargs, types=['common']):
    defaults = {}
    for key in types:
        if key == 'common':
            defaults['label'] = obj.GetTitle()
            defaults['visible'] = getattr(obj, 'visible', True)
            defaults['alpha'] = getattr(obj, 'alpha', None)
        elif key == 'line':
            defaults['linestyle'] = obj.GetLineStyle('mpl')
            defaults['linewidth'] = obj.GetLineWidth()
        elif key == 'fill':
            defaults['edgecolor'] = kwargs.get('color', obj.GetLineColor('mpl'))
            defaults['facecolor'] = kwargs.get('color', obj.GetFillColor('mpl'))
            root_fillstyle = obj.GetFillStyle('root')
            if root_fillstyle == 0:
                if not kwargs.get('fill'):
                    defaults['facecolor'] = 'none'
                defaults['fill'] = False
            elif root_fillstyle == 1001:
                defaults['fill'] = True
            else:
                defaults['hatch'] = obj.GetFillStyle('mpl')
                defaults['facecolor'] = 'none'
        elif key == 'marker':
            defaults['marker'] = obj.GetMarkerStyle('mpl')
            defaults['markersize'] = obj.GetMarkerSize() * 5
            defaults['markeredgecolor'] = obj.GetMarkerColor('mpl')
            defaults['markerfacecolor'] = obj.GetMarkerColor('mpl')
        elif key == 'errors':
            defaults['ecolor'] = obj.GetLineColor('mpl')
        elif key == 'errorbar':
            defaults['fmt'] = obj.GetMarkerStyle('mpl')
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


def _get_highest_zorder(axes):
    return max([c.get_zorder() for c in axes.get_children()])


def _maybe_reversed(x, reverse=False):
    if reverse:
        return reversed(x)
    return x


def hist(hists,
         stacked=True,
         reverse=False,
         xpadding=0, ypadding=.1,
         yerror_in_padding=True,
         logy=None,
         snap=True,
         axes=None,
         **kwargs):
    """
    Make a matplotlib hist plot from a ROOT histogram, stack or
    list of histograms.

    Parameters
    ----------

    hists : Hist, list of Hist, HistStack
        The histogram(s) to be plotted

    stacked : bool, optional (default=True)
        If True then stack the histograms with the first histogram on the
        bottom, otherwise overlay them with the first histogram in the
        background.

    reverse : bool, optional (default=False)
        If True then reverse the order of the stack or overlay.

    xpadding : float or 2-tuple of floats, optional (default=0)
        Padding to add on the left and right sides of the plot as a fraction of
        the axes width after the padding has been added. Specify unique left
        and right padding with a 2-tuple.

    ypadding : float or 2-tuple of floats, optional (default=.1)
        Padding to add on the top and bottom of the plot as a fraction of
        the axes height after the padding has been added. Specify unique top
        and bottom padding with a 2-tuple.

    yerror_in_padding : bool, optional (default=True)
        If True then make the padding inclusive of the y errors otherwise
        only pad around the y values.

    logy : bool, optional (default=None)
        Apply special treatment of a log-scale y-axis to display the histogram
        correctly. If None (the default) then automatically determine if the
        y-axis is log-scale.

    snap : bool, optional (default=True)
        If True (the default) then the origin is an implicit lower bound of the
        histogram unless the histogram has both positive and negative bins.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    kwargs : additional keyword arguments, optional
        All additional keyword arguments are passed to matplotlib's
        fill_between for the filled regions and matplotlib's step function
        for the edges.

    Returns
    -------

    The return value from matplotlib's hist function, or list of such return
    values if a stack or list of histograms was plotted.

    """
    if axes is None:
        axes = plt.gca()
    if logy is None:
        logy = axes.get_yscale() == 'log'
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    returns = []
    if isinstance(hists, _Hist):
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
            high.alpha = getattr(h, 'alpha', None)
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
        for h in _maybe_reversed(hists, reverse):
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
        zorder = _get_highest_zorder(axes) + 1
    _set_defaults(h, kwargs, ['common', 'line', 'fill'])
    kwargs_proxy = kwargs.copy()
    fill = kwargs.pop('fill', False) or ('hatch' in kwargs)
    if fill:
        # draw the fill without the edge
        if bottom is None:
            bottom = h.Clone()
            bottom.Reset()
        fill_between(bottom, h, axes=axes, logy=logy, linewidth=0,
                     facecolor=kwargs['facecolor'],
                     edgecolor=kwargs['edgecolor'],
                     hatch=kwargs.get('hatch', None),
                     alpha=kwargs['alpha'],
                     zorder=zorder)
    # draw the edge
    s = step(h, axes=axes, logy=logy, label=None,
         zorder=zorder + 1, alpha=kwargs['alpha'],
         color=kwargs.get('color'))
    # draw the legend proxy
    if getattr(h, 'legendstyle', '').upper() == 'F':
        proxy = plt.Rectangle((0, 0), 0, 0, **kwargs_proxy)
        axes.add_patch(proxy)
    else:
        # be sure the linewidth is greater than zero...
        proxy = plt.Line2D((0, 0), (0, 0),
                           linestyle=kwargs_proxy['linestyle'],
                           linewidth=kwargs_proxy['linewidth'],
                           color=kwargs_proxy['edgecolor'],
                           alpha=kwargs['alpha'],
                           label=kwargs_proxy['label'])
        axes.add_line(proxy)
    return proxy, s[0]


def bar(hists,
        stacked=True,
        reverse=False,
        xerr=False, yerr=True,
        xpadding=0, ypadding=.1,
        yerror_in_padding=True,
        rwidth=0.8,
        snap=True,
        axes=None,
        **kwargs):
    """
    Make a matplotlib bar plot from a ROOT histogram, stack or
    list of histograms.

    Parameters
    ----------

    hists : Hist, list of Hist, HistStack
        The histogram(s) to be plotted

    stacked : bool or string, optional (default=True)
        If True then stack the histograms with the first histogram on the
        bottom, otherwise overlay them with the first histogram in the
        background. If 'cluster', then the bars will be arranged side-by-side.

    reverse : bool, optional (default=False)
        If True then reverse the order of the stack or overlay.

    xerr : bool, optional (default=False)
        If True, x error bars will be displayed.

    yerr : bool or string, optional (default=True)
        If False, no y errors are displayed.  If True, an individual y
        error will be displayed for each hist in the stack.  If 'linear' or
        'quadratic', a single error bar will be displayed with either the
        linear or quadratic sum of the individual errors.

    xpadding : float or 2-tuple of floats, optional (default=0)
        Padding to add on the left and right sides of the plot as a fraction of
        the axes width after the padding has been added. Specify unique left
        and right padding with a 2-tuple.

    ypadding : float or 2-tuple of floats, optional (default=.1)
        Padding to add on the top and bottom of the plot as a fraction of
        the axes height after the padding has been added. Specify unique top
        and bottom padding with a 2-tuple.

    yerror_in_padding : bool, optional (default=True)
        If True then make the padding inclusive of the y errors otherwise
        only pad around the y values.

    rwidth : float, optional (default=0.8)
        The relative width of the bars as a fraction of the bin width.

    snap : bool, optional (default=True)
        If True (the default) then the origin is an implicit lower bound of the
        histogram unless the histogram has both positive and negative bins.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    kwargs : additional keyword arguments, optional
        All additional keyword arguments are passed to matplotlib's bar
        function.

    Returns
    -------

    The return value from matplotlib's bar function, or list of such return
    values if a stack or list of histograms was plotted.

    """
    if axes is None:
        axes = plt.gca()
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
    logy = kwargs.pop('log', axes.get_yscale() == 'log')
    kwargs['log'] = logy
    returns = []
    if isinstance(hists, _Hist):
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
        hlist = _maybe_reversed(hists, reverse)
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
        hlist = _maybe_reversed(hists, reverse)
        toterr = bottom = None
        if yerr == 'linear':
            toterr = [sum([h.GetBinError(i) for h in hists])
                      for i in range(1, hists[0].nbins(0) + 1)]
        elif yerr == 'quadratic':
            toterr = [sqrt(sum([h.GetBinError(i) ** 2 for h in hists]))
                      for i in range(1, hists[0].nbins(0) + 1)]
        for i, h in enumerate(hlist):
            err = None
            if yerr is True:
                err = True
            elif yerr and i == (nhists - 1):
                err = toterr
            returns.append(_bar(
                h,
                xerr=xerr, yerr=err,
                bottom=list(bottom.y()) if bottom else None,
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
    left = [h.xedgesl(i) + h.xwidth(i) * roffset
            for i in range(1, h.nbins(0) + 1)]
    height = list(h.y())
    return axes.bar(left, height, width=width, xerr=xerr, yerr=yerr, **kwargs)


def errorbar(hists,
             xerr=True, yerr=True,
             xpadding=0, ypadding=.1,
             xerror_in_padding=True,
             yerror_in_padding=True,
             emptybins=True,
             snap=True,
             axes=None,
             **kwargs):
    """
    Make a matplotlib errorbar plot from a ROOT histogram or graph
    or list of histograms and graphs.

    Parameters
    ----------

    hists : Hist, Graph or list of Hist and Graph
        The histogram(s) and/or Graph(s) to be plotted

    xerr : bool, optional (default=True)
        If True, x error bars will be displayed.

    yerr : bool or string, optional (default=True)
        If False, no y errors are displayed.  If True, an individual y
        error will be displayed for each hist in the stack.  If 'linear' or
        'quadratic', a single error bar will be displayed with either the
        linear or quadratic sum of the individual errors.

    xpadding : float or 2-tuple of floats, optional (default=0)
        Padding to add on the left and right sides of the plot as a fraction of
        the axes width after the padding has been added. Specify unique left
        and right padding with a 2-tuple.

    ypadding : float or 2-tuple of floats, optional (default=.1)
        Padding to add on the top and bottom of the plot as a fraction of
        the axes height after the padding has been added. Specify unique top
        and bottom padding with a 2-tuple.

    xerror_in_padding : bool, optional (default=True)
        If True then make the padding inclusive of the x errors otherwise
        only pad around the x values.

    yerror_in_padding : bool, optional (default=True)
        If True then make the padding inclusive of the y errors otherwise
        only pad around the y values.

    emptybins : bool, optional (default=True)
        If True (the default) then plot bins with zero content otherwise only
        show bins with nonzero content.

    snap : bool, optional (default=True)
        If True (the default) then the origin is an implicit lower bound of the
        histogram unless the histogram has both positive and negative bins.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    kwargs : additional keyword arguments, optional
        All additional keyword arguments are passed to matplotlib's errorbar
        function.

    Returns
    -------

    The return value from matplotlib's errorbar function, or list of such
    return values if a list of histograms and/or graphs was plotted.

    """
    if axes is None:
        axes = plt.gca()
    curr_xlim = axes.get_xlim()
    curr_ylim = axes.get_ylim()
    was_empty = not axes.has_data()
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
        returns = []
        for h in hists:
            returns.append(errorbar(
                h, xerr=xerr, yerr=yerr, axes=axes,
                xpadding=xpadding, ypadding=ypadding,
                xerror_in_padding=xerror_in_padding,
                yerror_in_padding=yerror_in_padding,
                snap=snap,
                emptybins=emptybins,
                **kwargs))
    return returns


def _errorbar(h, xerr, yerr, axes=None, emptybins=True, zorder=None, **kwargs):
    if axes is None:
        axes = plt.gca()
    if zorder is None:
        zorder = _get_highest_zorder(axes) + 1
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
        if xerr is not False and xerr is not None:
            xerr = xerr[:, nonempty]
        if yerr is not False and yerr is not None:
            yerr = yerr[:, nonempty]
    return axes.errorbar(x, y, xerr=xerr, yerr=yerr, zorder=zorder, **kwargs)


def step(h, logy=None, axes=None, **kwargs):
    """
    Make a matplotlib step plot from a ROOT histogram.

    Parameters
    ----------

    h : Hist
        A rootpy Hist

    logy : bool, optional (default=None)
        If True then clip the y range between 1E-300 and 1E300.
        If None (the default) then automatically determine if the axes are
        log-scale and if this clipping should be performed.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    kwargs : additional keyword arguments, optional
        Additional keyword arguments are passed directly to
        matplotlib's fill_between function.

    Returns
    -------

    Returns the value from matplotlib's fill_between function.

    """
    if axes is None:
        axes = plt.gca()
    if logy is None:
        logy = axes.get_yscale() == 'log'
    _set_defaults(h, kwargs, ['common', 'line'])
    if kwargs.get('color') is None:
        kwargs['color'] = h.GetLineColor('mpl')
    y = np.array(list(h.y()) + [0.])
    if logy:
        np.clip(y, 1E-300, 1E300, out=y)
    return axes.step(list(h.xedges()), y, where='post', **kwargs)


def fill_between(a, b, logy=None, axes=None, **kwargs):
    """
    Fill the region between two histograms or graphs.

    Parameters
    ----------

    a : Hist
        A rootpy Hist

    b : Hist
        A rootpy Hist

    logy : bool, optional (default=None)
        If True then clip the region between 1E-300 and 1E300.
        If None (the default) then automatically determine if the axes are
        log-scale and if this clipping should be performed.

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    kwargs : additional keyword arguments, optional
        Additional keyword arguments are passed directly to
        matplotlib's fill_between function.

    Returns
    -------

    Returns the value from matplotlib's fill_between function.

    """
    if axes is None:
        axes = plt.gca()
    if logy is None:
        logy = axes.get_yscale() == 'log'
    if not isinstance(a, _Hist) or not isinstance(b, _Hist):
        raise TypeError(
            "fill_between only operates on 1D histograms")
    a.check_compatibility(b, check_edges=True)
    x = []
    top = []
    bottom = []
    for abin, bbin in zip(a.bins(overflow=False), b.bins(overflow=False)):
        up = max(abin.value, bbin.value)
        dn = min(abin.value, bbin.value)
        x.extend([abin.x.low, abin.x.high])
        top.extend([up, up])
        bottom.extend([dn, dn])
    x = np.array(x)
    top = np.array(top)
    bottom = np.array(bottom)
    if logy:
        np.clip(top, 1E-300, 1E300, out=top)
        np.clip(bottom, 1E-300, 1E300, out=bottom)
    return axes.fill_between(x, top, bottom, **kwargs)


def hist2d(h, axes=None, colorbar=False, **kwargs):
    """
    Draw a 2D matplotlib histogram plot from a 2D ROOT histogram.

    Parameters
    ----------

    h : Hist2D
        A rootpy Hist2D

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    colorbar : Boolean, optional (default=False)
        If True, include a colorbar in the produced plot

    kwargs : additional keyword arguments, optional
        Additional keyword arguments are passed directly to
        matplotlib's hist2d function.

    Returns
    -------

    Returns the value from matplotlib's hist2d function.

    """
    if axes is None:
        axes = plt.gca()
    X, Y = np.meshgrid(list(h.x()), list(h.y()))
    x = X.ravel()
    y = Y.ravel()
    z = np.array(h.z()).T
    # returns of hist2d: (counts, xedges, yedges, Image)
    return_values = axes.hist2d(x, y, weights=z.ravel(),
                                bins=(list(h.xedges()), list(h.yedges())),
                                **kwargs)
    if colorbar:
        mappable = return_values[-1]
        plt.colorbar(mappable, ax=axes)
    return return_values


def imshow(h, axes=None, colorbar=False, **kwargs):
    """
    Draw a matplotlib imshow plot from a 2D ROOT histogram.

    Parameters
    ----------

    h : Hist2D
        A rootpy Hist2D

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    colorbar : Boolean, optional (default=False)
        If True, include a colorbar in the produced plot

    kwargs : additional keyword arguments, optional
        Additional keyword arguments are passed directly to
        matplotlib's imshow function.

    Returns
    -------

    Returns the value from matplotlib's imshow function.

    """
    kwargs.setdefault('aspect', 'auto')

    if axes is None:
        axes = plt.gca()
    z = np.array(h.z()).T

    axis_image= axes.imshow(
        z,
        extent=[
            h.xedges(1), h.xedges(h.nbins(0) + 1),
            h.yedges(1), h.yedges(h.nbins(1) + 1)],
        interpolation='nearest',
        origin='lower',
        **kwargs)
    if colorbar:
        plt.colorbar(axis_image, ax=axes)
    return axis_image


def contour(h, axes=None, zoom=None, label_contour=False, **kwargs):
    """
    Draw a matplotlib contour plot from a 2D ROOT histogram.

    Parameters
    ----------

    h : Hist2D
        A rootpy Hist2D

    axes : matplotlib Axes instance, optional (default=None)
        The axes to plot on. If None then use the global current axes.

    zoom : float or sequence, optional (default=None)
        The zoom factor along the axes. If a float, zoom is the same for each
        axis. If a sequence, zoom should contain one value for each axis.
        The histogram is zoomed using a cubic spline interpolation to create
        smooth contours.

    label_contour : Boolean, optional (default=False)
        If True, labels are printed on the contour lines.

    kwargs : additional keyword arguments, optional
        Additional keyword arguments are passed directly to
        matplotlib's contour function.

    Returns
    -------

    Returns the value from matplotlib's contour function.

    """
    if axes is None:
        axes = plt.gca()
    x = np.array(list(h.x()))
    y = np.array(list(h.y()))
    z = np.array(h.z()).T
    if zoom is not None:
        from scipy import ndimage
        if hasattr(zoom, '__iter__'):
            zoom = list(zoom)
            x = ndimage.zoom(x, zoom[0])
            y = ndimage.zoom(y, zoom[1])
        else:
            x = ndimage.zoom(x, zoom)
            y = ndimage.zoom(y, zoom)
        z = ndimage.zoom(z, zoom)
    return_values = axes.contour(x, y, z, **kwargs)
    if label_contour:
        plt.clabel(return_values)
    return return_values
