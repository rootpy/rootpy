import ROOT
from rootpy.plotting import Hist
from rootpy.plotting.style import use_style


def test_atlas():

    with use_style('ATLAS'):
        # generate some random data
        hpx = Hist(100, -4, 4, name="hpx", title="This is the px distribution")
        ROOT.gRandom.SetSeed()
        for i in xrange(25000):
            hpx.Fill(ROOT.gRandom.Gaus())
        hpx.GetXaxis().SetTitle("random variable [unit]")
        hpx.GetYaxis().SetTitle("#frac{dN}{dr} [unit^{-1}]")
        hpx.SetMaximum(1000.)
        hpx.Draw()

if __name__ == "__main__":
    import nose
    nose.runmodule()
