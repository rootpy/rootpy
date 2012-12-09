#!/usr/bin/env python
"""
==========================
Setting the plotting style
==========================

This example demonstrates how to set the plotting style.
"""
print __doc__
import ROOT
import rootpy
rootpy.log.basic_config_colorized()
from rootpy.plotting import Hist
from rootpy.plotting.style import get_style
from rootpy.interactive import wait


atlas_style = get_style('ATLAS')

# use styles as context managers
# the atlas style will only apply to the context within the following context
with atlas_style:
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
