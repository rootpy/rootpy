#!/usr/bin/env python
import numpy as np
from rootpy.plotting import Hist, HistStack
import rootpy.root2matplotlib as rplt
import matplotlib.pyplot as plt

# create normal distributions
mu1, mu2, sigma = 100, 140, 15
x1 = mu1 + sigma * np.random.randn(10000)
x2 = mu2 + sigma * np.random.randn(10000)

# create histograms
h1 = Hist(100, 40, 160)
h2 = Hist(100, 40, 160)

# fill the histograms with our distributions
map(h1.Fill, x1)
map(h2.Fill, x2)

# normalize the histograms
h1 /= h1.Integral()
h2 /= h2.Integral()

# set visual attributes
h1.SetFillStyle('solid')
h1.SetFillColor('green')
h1.SetLineColor('green')

h2.SetFillStyle('solid')
h2.SetFillColor('red')
h2.SetLineColor('red')

stack = HistStack()
stack.Add(h1)
stack.Add(h2)

# plot with ROOT
h1.SetTitle('Histogram of IQ: #mu=100, #sigma=15')
stack.Draw()
h1.GetXaxis().SetTitle('Smarts')
h1.GetYaxis().SetTitle('Probability')

# plot with matplotlib
rplt.hist(stack, alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.show()
