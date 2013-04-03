#!/usr/bin/env python
"""
==========================
Setting the plotting style
==========================

This example demonstrates how to set the plotting style.
"""
print __doc__
import sys
import ROOT
import rootpy
rootpy.log.basic_config_colorized()
from rootpy.plotting import Hist
from rootpy.plotting.style import get_style
from rootpy.interactive import wait

if len(sys.argv) == 1:
    print('You can also specify a style as argument')

try:
    style = get_style(sys.argv[1])
except:
    print('Invalid style: `{}`. Using default style `ATLAS`'.format(sys.argv[1]))
    style = get_style('ATLAS')

# Use styles as context managers. The ATLAS style will only apply
# within the following context:
with style:
    hpx = Hist(100, -4, 4, name="hpx", title="This is the px distribution")
    # generate some random data
    ROOT.gRandom.SetSeed()
    for i in xrange(25000):
        hpx.Fill(ROOT.gRandom.Gaus())
    hpx.GetXaxis().SetTitle("random variable [unit]")
    hpx.GetYaxis().SetTitle("#frac{dN}{dr} [unit^{-1}]")
    hpx.SetMaximum(1000.)
    hpx.Draw()
    wait()
