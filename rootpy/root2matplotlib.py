from .plotting.hist import _HistBase
from .plotting import HistStack
import matplotlib.pyplot as plt

def hist(h, **kwargs):
    
    if isinstance(h, _HistBase):
        return _hist(h, **kwargs)
    if hasattr(h, "__getitem__"):
        returns = []
        previous = None
        kwargs['histtype'] = 'bar'
        for histo in h:
            r = _hist(histo, bottom = previous, **kwargs)
            previous = r[0]
            returns.append(r)
        return returns

def _hist(h, **kwargs):

    fillstyle = h.GetFillStyle()
    hatch = fillstyle if fillstyle not in ["", "hollow", "solid"] else None
    defaults = {'facecolor' : h.GetFillColor(),
                'edgecolor' : h.GetLineColor(),
                'fill' : (fillstyle != "hollow"),
                'hatch' : hatch,
                'linestyle' : h.GetLineStyle(),
                'linewidth' : h.GetLineWidth(),
                'label' : h.GetTitle(),
                'visible' : h.visible,
                'histtype' : 'stepfilled',
                }
    for key, value in defaults.items():
        if not kwargs.has_key(key):
            kwargs[key] = value

    was_empty = plt.ylim()[1] == 1. # TODO there must be a better way to determine this...
    r = plt.hist(h.xcenters, weights = h, bins = h.xedges, **kwargs)
    if was_empty:
        plt.ylim(ymax = h.maximum()*1.1)
    else:
        ymin, ymax = plt.ylim()
        plt.ylim(ymax = max(ymax, h.maximum()*1.1))
    return r

def errorbar(h, **kwargs):

    defaults = {'color' : h.linecolor,
                'label' : h.GetTitle(),
                'visible' : h.visible,
                'fmt' : h.markerstyle,
                'capsize' : 0,
                }
    for key, value in defaults.items():
        if not kwargs.has_key(key):
            kwargs[key] = value

    return plt.errorbar(h.xcenters, h,
                        yerr=list(h.yerrors()),
                        xerr=list(h.xerrors()),
                        **kwargs)

