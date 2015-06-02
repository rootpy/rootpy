#!/usr/bin/env python
"""
========================================
Plot a 2D ROOT histogram with matplotlib
========================================

This example demonstrates how a 2D ROOT histogram can be displayed with
matplotlib.
"""
print(__doc__)
import ROOT
from matplotlib import pyplot as plt
from rootpy.plotting import root2matplotlib as rplt
from rootpy.plotting import Hist2D
import numpy as np

a = Hist2D(100, -3, 3, 100, 0, 6)
a.fill_array(np.random.multivariate_normal(
    mean=(0, 3),
    cov=[[1, .5], [.5, 1]],
    size=(1E6,)))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax1.set_title('hist2d')
rplt.hist2d(a, axes=ax1)

ax2.set_title('imshow')
im = rplt.imshow(a, axes=ax2)

ax3.set_title('contour')
rplt.contour(a, axes=ax3)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

if not ROOT.gROOT.IsBatch():
    plt.show()
