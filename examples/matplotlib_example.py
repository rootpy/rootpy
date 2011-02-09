#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from rootpy.plotting import *

# create a normal distribution
mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# create a histogram with 100 bins from 40 to 160
h = Hist(100,40,160)

# fill
map(h.Fill, x)

# normalize
h /= h.Integral()

# plot with ROOT
h.GetXaxis().SetTitle('Smarts')
h.GetYaxis().SetTitle('Probability')
h.SetTitle("Histogram of IQ: #mu=100, #sigma=15")
h.Draw("hist")

# plot with matplotlib
plt.hist(h.xcenters, weights = h, bins = h.xedges, facecolor='green', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.show()
