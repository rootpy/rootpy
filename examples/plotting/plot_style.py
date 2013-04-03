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

try:
    style_name = sys.argv[1]
except IndexError:
    print('Specify the style with the style name as an argument')
    print('Using the ATLAS style')
    style_name = 'ATLAS'

try:
    style = get_style(style_name)
except ValueError:
    print('Invalid style: `{}`. Using the `ATLAS` style.'.format(style_name))
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
