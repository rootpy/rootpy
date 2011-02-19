from rootpy.plotting import _HistBase, HistStack
import matplotlib.pyplot as plt

def hist(h, **kwargs):
    
    if isinstance(h, _HistBase):
        return _hist(h, **kwargs)
    if hasattr(h, "__getitem__"):
        returns = []
        kwargs['histtype'] = 'bar'     
        previous = None
        for histo in h:
            r = _hist(histo, bottom = previous, **kwargs)
            previous = r[0]
            returns.append(r)
        return returns

def _hist(h, **kwargs):

    fillstyle = h.GetFillStyle()
    if not kwargs.has_key('facecolor'):
        kwargs['facecolor'] = h.GetFillColor()
    if not kwargs.has_key('edgecolor'):
        kwargs['edgecolor'] = h.GetLineColor()
    if not kwargs.has_key('fill'):
        kwargs['fill'] = (fillstyle != "hollow")
    if not kwargs.has_key('hatch'):
        kwargs['hatch'] = (fillstyle if fillstyle not in ["", "hollow", "solid"] else None)
    if not kwargs.has_key('linestyle'):
        kwargs['linestyle'] = h.GetLineStyle()
    if not kwargs.has_key('linewidth'):
        kwargs['linewidth'] = h.GetLineWidth()
    if not kwargs.has_key('label'):
        kwargs['label'] = h.GetTitle()
    if not kwargs.has_key('visible'):
        kwargs['visible'] = h.visible
    if not kwargs.has_key('histtype'):
        kwargs['histtype'] = 'stepfilled'

    return plt.hist(h.xcenters, weights = h, bins = h.xedges, **kwargs)
