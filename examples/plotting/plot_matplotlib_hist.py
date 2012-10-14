#!/usr/bin/env python
"""
=====================================
Plot a ROOT histogram with matplotlib
=====================================

This example demonstrates how a ROOT histogram can be styled with simple
attributes and displayed via ROOT or matplotlib.
"""
print __doc__
import numpy as np
from rootpy.plotting import Hist, HistStack, Legend, Canvas
import rootpy.plotting.root2matplotlib as rplt
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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
h3.markersize=1.2

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
canvas = Canvas(width=700, height=500)
canvas.SetLeftMargin(0.15)
canvas.SetBottomMargin(0.15)
canvas.SetTopMargin(0.05)
canvas.SetRightMargin(0.05)
stack.Draw()
h3.Draw('E1 same')
stack.xaxis.SetTitle('Mass')
stack.yaxis.SetTitle('Events')
legend = Legend(2)
legend.AddEntry(h1, 'F')
legend.AddEntry(h2, 'F')
legend.AddEntry(h3, 'P')
legend.Draw()
canvas.Modified()
canvas.Update()

# plot with matplotlib
fig = plt.figure(figsize=(7, 5), dpi=100, facecolor='white')
axes = plt.axes([0.15, 0.15, 0.8, 0.8])
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.yaxis.set_minor_locator(AutoMinorLocator())
axes.tick_params(which='major', labelsize=15, length=8)
axes.tick_params(which='minor', length=4)
rplt.bar(stack, stacked=True, axes=axes)
rplt.errorbar(h3, xerr=False, emptybins=False, axes=axes)
plt.xlabel('Mass', position=(1., 0.), ha='right')
plt.ylabel('Events', position=(0., 1.), va='top')
plt.legend(numpoints=1)
plt.show()
