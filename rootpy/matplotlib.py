
import matplotlib.pyplot as plt

def hist(hist, **kwargs):

    fillstyle = hist.GetFillStyle()
    return plt.hist(hist.xcenters, weights = hist, bins = hist.xedges,
        facecolor = hist.GetFillColor(),
        edgecolor = hist.GetLineColor(),
        fill = (fillstyle != "hollow"),
        hatch = (fillstyle if fillstyle not in ["", "hollow", "solid"] else None),
        label = hist.GetTitle(),
        visible = hist.visible,
        **kwargs)
