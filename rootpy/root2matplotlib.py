
import matplotlib.pyplot as plt

def hist(h, **kwargs):

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

    return plt.hist(h.xcenters, weights = h, bins = h.xedges, **kwargs)
