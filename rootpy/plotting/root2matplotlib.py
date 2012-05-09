from .hist import _HistBase, HistStack
from .graph import Graph
from math import sqrt
import matplotlib.pyplot as plt

__all__ = [
    'hist',
    'bar',
    'errorbar',
]


def _set_defaults(h, kwargs, types=['common']):
    defaults = {}
    for key in types:
        if key == 'common':
            defaults['label'] = h.GetTitle()
            defaults['visible'] = h.visible
        elif key == 'fill':
            defaults['linestyle'] = h.GetLineStyle('mpl')
            defaults['facecolor'] = h.GetFillColor('mpl')
            defaults['hatch'] = h.GetFillStyle('mpl')
            defaults['facecolor'] = h.GetFillColor('mpl')
            defaults['edgecolor'] = h.GetLineColor('mpl')
        elif key == 'errors':
            defaults['ecolor'] = h.GetLineColor('mpl')
            defaults['color'] = h.GetMarkerColor('mpl')
            defaults['fmt'] = h.GetMarkerStyle('mpl')
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value


def _set_bounds(h, was_empty):

    if was_empty:
        plt.ylim(ymax=h.maximum() * 1.1)
        plt.xlim([h.xedgesl(0), h.xedgesh(-1)])
    else:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax=max(ymax, h.maximum() * 1.1))
        xmin, xmax = plt.xlim()
        plt.xlim([min(xmin, h.xedgesl(0)), max(xmax, h.xedgesh(-1))])


def maybe_reversed(x, reverse=False):
    if reverse:
        return reversed(x)
    return x


def hist(hists, stacked=True, reverse=False, axes=None, **kwargs):
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
        _set_bounds(hists, was_empty)
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
            axes.hist(hsum.x, weights=hsum, bins=hsum.xedges(),
                      histtype='step', edgecolor=hsum.GetLineColor())
        _set_bounds(sum(hists), was_empty)
    else:
        for h in maybe_reversed(hists, reverse):
            returns.append(_hist(h, axes=axes, **kwargs))
        _set_bounds(max(hists), was_empty)
    return returns


def _hist(h, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'fill'])
    kwargs['histtype'] = h.GetFillStyle('root') and 'stepfilled' or 'step'
    return axes.hist(list(h.x()), weights=list(h.y()), bins=list(h.xedges()), **kwargs)


def bar(hists, stacked=True, reverse=False,
        yerr=False, rwidth=0.8, axes=None, **kwargs):
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
        _set_bounds(hists, was_empty)
    elif stacked == 'cluster':
        hlist = maybe_reversed(hists, reverse)
        contents = [list(h) for h in hlist]
        xcenters = [h.x() for h in hlist]
        for i, h in enumerate(hlist):
            width = rwidth/nhists
            offset = (1 - rwidth) / 2 + i * width
            returns.append(_bar(h, offset, width, yerr, axes=axes, **kwargs))
        _set_bounds(sum(hists), was_empty)
    elif stacked is True:
        hlist = maybe_reversed(hists, reverse)
        bottom, toterr = None, None
        if yerr == 'linear':
            toterr = [sum([h.GetBinError(i + 1) for h in hists])
                      for i in range(len(hists[0]))]
        elif yerr == 'quadratic':
            toterr = [sqrt(sum([h.GetBinError(i + 1)**2 for h in hists]))
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
                bottom = h
        _set_bounds(max(hists), was_empty)
    else:
        for h in hlist:
            returns.append(_bar(h, yerr=bool(yerr), axes=axes, **kwargs))
        _set_bounds(max(hists), was_empty)
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


def errorbar(hists, xerr=True, yerr=True, axes=None, **kwargs):
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
        returns = _errorbar(hists, xerr, yerr, axes=axes, **kwargs)
        _set_bounds(hists, was_empty)
    else:
        for h in hists:
            returns.append(_errorbar(h, xerr, yerr, axes=axes, **kwargs))
        _set_bounds(max(hists), was_empty)
    return returns


def _errorbar(h, xerr, yerr, axes=None, **kwargs):

    if axes is None:
        axes = plt.gca()
    _set_defaults(h, kwargs, ['common', 'errors'])
    if xerr:
        xerr = [list(h.xerrl()), list(h.xerrh())]
    if yerr:
        yerr = [list(h.yerrl()), list(h.yerrh())]
    return axes.errorbar(list(h.x()), list(h.y()), xerr=xerr, yerr=yerr, **kwargs)
