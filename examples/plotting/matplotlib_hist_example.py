#!/usr/bin/env python
import numpy as np
from rootpy.plotting import Hist, Legend, Canvas
import rootpy.root2matplotlib as rplt
import matplotlib.pyplot as plt

# create a normal distribution
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# create a histogram with 100 bins from 40 to 160
h = Hist(100, 40, 160)

# fill the histogram with our distribution
map(h.Fill, x)

# normalize the histogram
h /= h.Integral()

# set visual attributes
h.SetFillStyle('/')
h.SetFillColor('green')
h.SetLineColor('green')

# plot with ROOT
c = Canvas(width=800, height=600)
h.GetXaxis().SetTitle('Smarts')
h.GetYaxis().SetTitle('Probability')
h.SetTitle("Histogram of IQ: #mu=100, #sigma=15")
h.Draw("hist")
legend = Legend(1)
legend.AddEntry(h, 'F')
legend.Draw()
c.SaveAs('root_hist.png')

# plot with matplotlib
plt.figure(figsize=(8, 6), dpi=100)
rplt.hist(h, label=r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.legend()
plt.savefig('matplotlib_hist.png')
plt.show()
