# Copyright 2013 the rootpy developers
# distributed under the terms of the GNU General Public License

import ROOT
from rootpy.plotting import Hist
from rootpy.plotting.style import get_style
from rootpy.plotting.style.lhcb.labels import lhcb_label


def test_lhcb():
    style = get_style('LHCb')

    with style:
        hpx = Hist(100, -4, 4, name="hpx", title="This is the px distribution")
        ROOT.gRandom.SetSeed()

        for i in xrange(1000):
            hpx.Fill(ROOT.gRandom.Gaus())

        hpx.GetXaxis().SetTitle("random variable [unit]")
        hpx.GetYaxis().SetTitle("#frac{dN}{dr} [unit^{-1}]")
        hpx.SetMaximum(80.)
        hpx.Draw()

        lhcb_label("R", "preliminary")


if __name__ == "__main__":
    import nose
    nose.runmodule()
