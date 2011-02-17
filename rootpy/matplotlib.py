
import matplotlib.pyplot as plt

def hist(hist, **kwargs):

   plt.hist(hist.xcenters, weights = hist, bins = hist.xedges, **kwargs)
