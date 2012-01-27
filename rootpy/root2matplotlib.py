from .plotting.hist import _HistBase
from .plotting import HistStack
from math import sqrt
import matplotlib.pyplot as plt

__all__ = [
    'hist',
    'histstack',
    'bar',
    'barstack',
    'errorbar',
]


def _set_defaults(h, kwargs, types=['common']):
    defaults = {}
    for key in types:
        if key == 'common':
            defaults['facecolor'] = h.GetFillColor('mpl')
            defaults['edgecolor'] = h.GetLineColor('mpl')
            defaults['linestyle'] = h.GetLineStyle('mpl')
            defaults['ecolor'] = h.GetMarkerColor('mpl')
            defaults['label'] = h.GetTitle()
            defaults['visible'] = h.visible
        elif key == 'fill':
            defaults['facecolor'] = h.GetFillColor('mpl')
            defaults['hatch'] = h.GetFillStyle('mpl')
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value


def _set_bounds(h, was_empty):

    if was_empty:
        plt.ylim(ymax=h.maximum() * 1.1)
        plt.xlim([h.xedges[0], h.xedges[-1]])
    else:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax=max(ymax, h.maximum() * 1.1))
        xmin, xmax = plt.xlim()
        plt.xlim([min(xmin, h.xedges[0]), max(xmax, h.xedges[-1])])


def maybe_reversed(x, reverse=False):
    if reverse:
        return reversed(x)
    return x


def hist(hists, stacked=True, reverse=False, **kwargs):
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
    if isinstance(hists, _HistBase):
        # This is a single histogram.
        returns = _hist(hists, **kwargs)
    elif stacked:
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
            plt.hist(hsum.xcenters, weights=hsum, bins=hsum.xedges,
                     histtype='step', edgecolor=hsum.GetLineColor())
    else:
        for h in maybe_reversed(hists, reverse):
            returns.append(_hist(h, **kwargs))
    _set_bounds(max(hists), was_empty)
    return returns


def _hist(h, **kwargs):

    _set_defaults(h, kwargs, ['common', 'fill'])
    kwargs['histtype'] = h.GetFillStyle('root') and 'stepfilled' or 'step'
    return plt.hist(h.xcenters, weights=h, bins=h.xedges, **kwargs)


def bar(hists, stacked=True, reverse=False, yerr=False, rwidth=0.8, **kwargs):
    """
    Make a matplotlib bar plot.

    *hists* may be a single :class:`rootpy.plotting.hist.Hist` object or a
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
        returns = _bar(hists, yerr, **kwargs)
    elif stacked == 'cluster':
        hlist = maybe_reversed(hists, reverse)
        contents = [list(h) for h in hlist]
        xcenters = [h.xcenters for h in hlist]
        for i, h in enumerate(hlist):
            width = rwidth/nhists
            offset = (1 - rwidth) / 2 + i * width
            returns.append(_bar(h, offset, width, yerr, **kwargs))
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
            returns.append(_bar(h, yerr=err, bottom=bottom, **kwargs))
            if bottom:
                bottom += h
            else:
                bottom = h
    else:
        for h in hlist:
            returns.append(_bar(h, yerr=bool(yerr), **kwargs))
    _set_bounds(sum(hists), was_empty)
    return returns


def _bar(h, roffset=0., rwidth=1., yerr=None, **kwargs):

    if yerr is True:
        yerr = list(h.yerrors())
    _set_defaults(h, kwargs)
    width = [x * rwidth for x in h.xwidths]
    left = [h.xedges[i] + h.xwidths[i] * roffset for i in range(len(h))]
    height = h
    return plt.bar(left, height, width, yerr=yerr, **kwargs)

def _errorbar(h, **kwargs):

    _set_defaults(h, kwargs)
    return plt.bar(left=h.xedges[:-1], height=h, width=h.xwidths, **kwargs)


def errorbar(h, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    defaults = {'color': h.linecolor,
                'label': h.GetTitle(),
                'visible': h.visible,
                'fmt': h.markerstyle,
                'capsize': 0,
                'label': h.GetTitle(),
                }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    r = plt.errorbar(h.xcenters, h,
                     yerr=list(h.yerrors()),
                     xerr=list(h.xerrors()),
                     **kwargs)
    _set_bounds(h, was_empty)
    return r
