#!/usr/bin/env python
import numpy as np
from rootpy.plotting import Hist
import rootpy.root2matplotlib as rplt
import matplotlib.pyplot as plt

# create a normal distribution
mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# create a histogram with 100 bins from 40 to 160
h = Hist(100,40,160)

# fill the histogram with our distribution
map(h.Fill, x)

# normalize the histogram
h /= h.Integral()

# set visual attributes
h.SetFillStyle("O")
h.SetFillColor("green")
h.SetLineColor("green")

# plot with ROOT
h.GetXaxis().SetTitle('Smarts')
h.GetYaxis().SetTitle('Probability')
h.SetTitle("Histogram of IQ: #mu=100, #sigma=15")
h.Draw("hist")

# plot with matplotlib
rplt.hist(h, alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.show()
