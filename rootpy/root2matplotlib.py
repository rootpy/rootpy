
import matplotlib.pyplot as plt

def hist(h, **kwargs):

    fillstyle = h.GetFillStyle()
    return plt.hist(h.xcenters, weights = h, bins = h.xedges,
        facecolor = h.GetFillColor(),
        edgecolor = h.GetLineColor(),
        fill = (fillstyle != "hollow"),
        hatch = (fillstyle if fillstyle not in ["", "hollow", "solid"] else None),
        label = h.GetTitle(),
        visible = h.visible,
        **kwargs)
