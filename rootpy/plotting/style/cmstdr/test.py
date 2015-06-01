# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from rootpy.plotting import Canvas, Hist
from rootpy.plotting.style import get_style
from rootpy.plotting.style.cmstdr.labels import CMS_label
from rootpy.interactive import wait

INTERACTIVE = False

"""
def test_cmstdr():
    style = get_style('CMSTDR')
    with style:
        canvas = Canvas()
        hpx = Hist(100, -4, 4, name="hpx", title="This is the px distribution")
        ROOT.gRandom.SetSeed()
        for i in range(1000):
            hpx.Fill(ROOT.gRandom.Gaus())
        hpx.GetXaxis().SetTitle("random variable [unit]")
        hpx.GetYaxis().SetTitle("#frac{dN}{dr} [unit^{-1}]")
        hpx.SetMaximum(100.)
        hpx.Draw()
        CMS_label("Testing 2050", sqrts=100)
        if INTERACTIVE:
            wait()
"""

if __name__ == "__main__":
    import nose
    INTERACTIVE = True
    nose.runmodule()
