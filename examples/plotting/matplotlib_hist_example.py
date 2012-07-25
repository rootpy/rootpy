#!/usr/bin/env python
import numpy as np
from rootpy.plotting import Hist, HistStack, Legend
import rootpy.plotting.root2matplotlib as rplt
import matplotlib.pyplot as plt

import ROOT
# Setting this to True (default in rootpy)
# changes how the histograms look in ROOT...
ROOT.TH1.SetDefaultSumw2(False)

# create normal distributions
mu1, mu2, sigma1, sigma2 = 100, 140, 15, 5
x1 = mu1 + sigma1 * np.random.randn(10000)
x2 = mu2 + sigma2 * np.random.randn(1000)
x1_obs = mu1 + sigma1 * np.random.randn(10000)
x2_obs = mu2 + sigma2 * np.random.randn(1000)

# create histograms
h1 = Hist(100, 40, 200, title='Background')
h2 = h1.Clone(title='Signal')
h3 = h1.Clone(title='Data')

# fill the histograms with our distributions
map(h1.Fill, x1)
map(h2.Fill, x2)
map(h3.Fill, x1_obs)
map(h3.Fill, x2_obs)

# set visual attributes
h1.fillstyle = 'solid'
h1.fillcolor = 'green'
h1.linecolor = 'green'
h1.linewidth = 0

h2.fillstyle = 'solid'
h2.fillcolor = 'red'
h2.linecolor = 'red'
h2.linewidth = 0

stack = HistStack()
stack.Add(h1)
stack.Add(h2)

# plot with ROOT
stack.Draw()
h3.Draw('E1 same')
stack.xaxis.SetTitle('Mass')
stack.yaxis.SetTitle('Events')
legend = Legend(2)
legend.AddEntry(h1, 'F')
legend.AddEntry(h2, 'F')
legend.AddEntry(h3, 'P')
legend.Draw()

# plot with matplotlib
plt.figure(facecolor='white')
rplt.bar(stack, stacked=True)
rplt.errorbar(h3, emptybins=False)
plt.xlabel('Mass', position=(1., 0.), ha='right')
plt.ylabel('Events', position=(0., 1.), va='top')
plt.legend(numpoints=1)
plt.show()
