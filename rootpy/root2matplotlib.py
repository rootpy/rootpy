from .plotting.hist import _HistBase
from .plotting import HistStack
import matplotlib.pyplot as plt


__all__ = [
    'hist',
    'errorbar',
]


def _set_bounds(h, was_empty):

    if was_empty:
        plt.ylim(ymax=h.maximum() * 1.1)
    else:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax=max(ymax, h.maximum() * 1.1))


def hist(h, **kwargs):

    was_empty = plt.ylim()[1] == 1.
    if isinstance(h, _HistBase):
        r = _hist(h, **kwargs)
        _set_bounds(h, was_empty)
        return r
    if hasattr(h, "__getitem__"):
        returns = []
        previous = None
        kwargs['histtype'] = 'bar'
        for histo in h:
            r = _hist(histo, bottom=previous, **kwargs)
            if previous is not None:
                previous = previous + histo
            else:
                previous = histo
            returns.append(r)
        _set_bounds(sum(h), was_empty)
        return returns


def _hist(h, **kwargs):

    defaults = {'facecolor': h.GetFillColor(),
                'edgecolor': h.GetLineColor(),
                #'fill' : (h.fillstyle != "hollow"),
                'hatch': h.fillstylempl,
                'linestyle': h.linestylempl,
                #'linewidth' : h.linewidthmpl,
                'label': h.GetTitle(),
                'visible': h.visible,
                'histtype': 'stepfilled',
                }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    return plt.hist(h.xcenters, weights=h, bins=h.xedges, **kwargs)


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
