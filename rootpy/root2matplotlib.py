from .plotting.hist import _HistBase
from .plotting import HistStack
import matplotlib.pyplot as plt


__all__ = [
    'hist',
    'histstack',
    'bar',
    'barstack',
    'errorbar',
]


def _set_defaults(h, kwargs):

    defaults = {'facecolor': h.GetFillColor(),
                'edgecolor': h.GetLineColor(),
                #'fill' : (h.fillstyle != "hollow"),
                'hatch': h.fillstylempl,
                'linestyle': h.linestylempl,
                #'linewidth' : h.linewidthmpl,
                'label': h.GetTitle(),
                'visible': h.visible,
                }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value


def _set_bounds(h, was_empty):

    if was_empty:
        plt.ylim(ymax=h.maximum() * 1.1)
    else:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax=max(ymax, h.maximum() * 1.1))


def hist(h, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    r = _hist(h, **kwargs)
    _set_bounds(h, was_empty)
    return r


def histstack(hists, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    returns = []
    previous = None
    kwargs['histtype'] = 'bar'
    for h in hists:
        r = _hist(h, bottom=previous, **kwargs)
        if previous is not None:
            previous = previous + h
        else:
            previous = h
        returns.append(r)
    _set_bounds(sum(hists), was_empty)
    return returns


def _hist(h, **kwargs):

    _set_defaults(h, kwargs)
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'stepfilled'
    return plt.hist(h.xcenters, weights=h, bins=h.xedges, **kwargs)


def bar(h, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    r = _bar(h, **kwargs)
    _set_bounds(h, was_empty)
    return r


def barstack(hists, show_errors=True, errors_on_top=True, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    returns = []
    previous = None
    for i, h in enumerate(hists):
        yerr = None
        if show_errors:
            if errors_on_top and i == len(hists) - 1:
                yerr = list(sum(hists).yerrors())
            elif not errors_on_top:
                yerr = list(h.yerrors())
        r = _bar(h, bottom=previous, yerr=yerr, **kwargs)
        if previous is not None:
            previous = previous + h
        else:
            previous = h
        returns.append(r)
    _set_bounds(sum(hists), was_empty)
    return returns


def _bar(h, **kwargs):

    _set_defaults(h, kwargs)
    return plt.bar(left=h.xedges[:-1], height=h, width=h.xwidths, **kwargs)


def errorbar(h, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    defaults = {'color': h.linecolor,
                'label': h.GetTitle(),
                'visible': h.visible,
                'fmt': h.markerstylempl,
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
