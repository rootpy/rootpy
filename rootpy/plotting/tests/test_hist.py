# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from random import gauss, uniform
from rootpy import ROOTVersion, ROOT_VERSION
from rootpy.plotting import Hist, Hist2D, Hist3D, HistStack, Efficiency, Graph
from rootpy.plotting import F2, F3
from rootpy.utils.extras import LengthMismatch
from rootpy.extern.six.moves import range
from nose.tools import (raises, assert_equal, assert_almost_equal,
                        assert_raises, assert_true, assert_false)


def test_init():
    # constructor arguments are repetitions of #bins, left bound, right bound.
    h2d = Hist2D(10, 0, 1, 50, -40, 10, name='2d hist')
    h3d = Hist3D(3, -1, 4, 10, -1000, -200, 2, 0, 1, name='3d hist')

    # variable-width bins may be created by passing the bin edges directly:
    h1d_variable = Hist([1, 4, 10, 100])
    h2d_variable = Hist2D([2, 4, 7, 100, 200], [-100, -50, 0, 10, 20])
    h3d_variable = Hist3D([1, 3, 10], [20, 50, 100], [-10, -5, 10, 20])

    # variable-width and constant-width bins can be mixed:
    h2d_mixed = Hist2D([2, 10, 30], 10, 1, 5)


def test_init_from_graph():
    hist = Hist(10, 0, 1)
    hist.FillRandom('gaus')
    graph = Graph(hist)
    hist2 = Hist(graph)
    # TODO
    #assert_almost_equal(list(hist.xedges()), list(hist2.xedges()))
    #assert_almost_equal(list(hist.y()), list(hist2.y()))
    #assert_almost_equal(list(hist.yerr()), list(hist2.yerr()))


@raises(ValueError)
def test_init_edge_order():
    # bin edges must be in ascending order
    Hist2D([10, 2, 30], 10, 1, 5)


@raises(ValueError)
def test_init_edge_repeated():
    # bin edges must not be repeated
    Hist([10, 10, 30])


def test_edges():
    h = Hist([1, 2, 3, 4])
    assert_equal(list(h.xedges()), [1, 2, 3, 4])
    assert_equal(list(h.xedges(overflow=True)),
                 [float('-inf'), 1, 2, 3, 4, float('inf')])
    assert_equal(h.xedges(0), float('-inf'))
    assert_equal(h.xedges(-1), float('inf'))
    assert_equal(h.xedges(5), float('inf'))
    # wrap around
    assert_equal(h.xedges(6), float('-inf'))
    for i in range(1, h.nbins() + 1):
        assert_equal(h.xedges(i), i)


def test_edgesl():
    h = Hist([1, 2, 3, 4])
    assert_equal(list(h.xedgesl()), [1, 2, 3])
    assert_equal(list(h.xedgesl(overflow=True)),
                 [float('-inf'), 1, 2, 3, 4])
    assert_equal(h.xedgesl(0), float('-inf'))
    assert_equal(h.xedgesl(-1), 4)
    assert_equal(h.xedgesl(4), 4)
    # wrap around
    assert_equal(h.xedgesl(5), float('-inf'))
    for i in range(1, h.nbins()):
        assert_equal(h.xedgesl(i), i)


def test_edgesh():
    h = Hist([1, 2, 3, 4])
    assert_equal(list(h.xedgesh()), [2, 3, 4])
    assert_equal(list(h.xedgesh(overflow=True)),
                 [1, 2, 3, 4, float('inf')])
    assert_equal(h.xedgesh(0), 1)
    assert_equal(h.xedgesh(-1), float('inf'))
    assert_equal(h.xedgesh(4), float('inf'))
    # wrap around
    assert_equal(h.xedgesh(5), 1)
    for i in range(1, h.nbins()):
        assert_equal(h.xedgesh(i), i + 1)


def test_width():
    h = Hist([1, 2, 4, 8])
    assert_equal(list(h.xwidth()), [1, 2, 4])
    assert_equal(list(h.xwidth(overflow=True)),
                 [float('inf'), 1, 2, 4, float('inf')])


def test_bounds():
    h = Hist(10, 0, 1)
    assert_equal(h.bounds(), (0, 1))
    h = Hist2D(10, 0, 1, 10, 1, 2)
    assert_equal(h.bounds(axis=0), (0, 1))
    assert_equal(h.bounds(axis=1), (1, 2))
    h = Hist3D(10, 0, 1, 10, 1, 2, 10, 2, 3)
    assert_equal(h.bounds(axis=0), (0, 1))
    assert_equal(h.bounds(axis=1), (1, 2))
    assert_equal(h.bounds(axis=2), (2, 3))


def test_ravel():
    hist = Hist2D(3, 0, 1, 4, 0, 1)
    for i, bin in enumerate(hist.bins()):
        bin.value = i
        bin.error = i
    rhist = hist.ravel()
    assert_equal(list(rhist.y()), list(range(12)))
    assert_equal(list(rhist.yerrh()), list(range(12)))


def test_uniform():
    hist = Hist(10, 0, 1)
    assert_true(hist.uniform())
    hist = Hist2D(10, 0, 1, [1, 10, 100])
    assert_false(hist.uniform())
    assert_true(hist.uniform(axis=0))


def test_stack():
    stack = HistStack()
    assert_equal(len(stack), 0)
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='red'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='blue'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='green'))
    assert_equal(len(stack), 3)
    stack2 = stack.Clone()
    assert_equal(stack2[2].linecolor, 'green')

    # test stacked=True
    a = Hist(2, 0, 1)
    b = Hist(2, 0, 1)
    a.Fill(0.2)
    b.Fill(0.2)
    b.Fill(0.8, 5)
    stack = HistStack([a, b])
    assert_equal(stack.min(), 2)
    assert_equal(stack.max(), 5)

    # test stacked=False
    a = Hist(2, 0, 20)
    b = Hist(2, 10, 20)  # binning may be different
    a.Fill(0.2)
    b.Fill(15, 5)
    stack = HistStack([a, b], stacked=False)
    assert_equal(stack.min(), 0)
    assert_equal(stack.max(), 5)


def test_indexing():
    hist = Hist(10, 0, 1)
    hist.Fill(0.5)
    assert_equal(hist[6].value, 1)
    assert_equal(hist[10].value, 0)
    assert_raises(IndexError, hist.__getitem__, -13)
    assert_raises(IndexError, hist.__getitem__, 12)


def test_slice_assign():
    hist = Hist(10, 0, 1)
    hist[:] = [i for i in range(len(hist))]
    assert all([a.value == b for a, b in zip(hist, range(len(hist)))])
    clone = hist.Clone()
    # reverse bins
    hist[:] = clone[::-1]
    assert all([a.value == b.value for a, b in zip(hist, clone[::-1])])


@raises(LengthMismatch)
def test_slice_assign_bad():
    hist = Hist(10, 0, 1)
    hist[:] = range(len(hist) + 1)


def test_overflow_underflow():
    h1d = Hist(10, 0, 1)
    h1d.Fill(-1)
    h1d.Fill(2)
    assert_equal(h1d.underflow(), 1)
    assert_equal(h1d.overflow(), 1)

    h2d = Hist2D(10, 0, 1, 10, 0, 1)
    h2d.Fill(-1, .5)
    h2d.Fill(2, .5)
    assert_equal(h2d.underflow()[h2d.axis(1).FindBin(.5)], 1)
    assert_equal(h2d.overflow()[h2d.axis(1).FindBin(.5)], 1)
    h2d.Fill(.5, -1)
    h2d.Fill(.5, 2)
    assert_equal(h2d.underflow(axis=1)[h2d.axis(0).FindBin(.5)], 1)
    assert_equal(h2d.overflow(axis=1)[h2d.axis(0).FindBin(.5)], 1)

    h3d = Hist3D(10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3d.Fill(-1, .5, .5)
    h3d.Fill(2, .5, .5)
    assert_equal(h3d.underflow()[h3d.axis(1).FindBin(.5)][h3d.axis(2).FindBin(.5)], 1)
    assert_equal(h3d.overflow()[h3d.axis(1).FindBin(.5)][h3d.axis(2).FindBin(.5)], 1)
    h3d.Fill(.5, -1, .5)
    h3d.Fill(.5, 2, .5)
    assert_equal(h3d.underflow(axis=1)[h3d.axis(0).FindBin(.5)][h3d.axis(2).FindBin(.5)], 1)
    assert_equal(h3d.overflow(axis=1)[h3d.axis(0).FindBin(.5)][h3d.axis(2).FindBin(.5)], 1)
    h3d.Fill(.5, .5, -1)
    h3d.Fill(.5, .5, 2)
    assert_equal(h3d.underflow(axis=2)[h3d.axis(0).FindBin(.5)][h3d.axis(1).FindBin(.5)], 1)
    assert_equal(h3d.overflow(axis=2)[h3d.axis(0).FindBin(.5)][h3d.axis(1).FindBin(.5)], 1)


def test_merge_bins():
    h1d = Hist(10, 0, 1)
    h1d.FillRandom('gaus', 1000)
    h1d_merged = h1d.merge_bins([(0, -1)])
    assert_equal(h1d_merged.nbins(0), 1)

    h3d = Hist3D(10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3d.FillRandom('gaus')
    h3d_merged = h3d.merge_bins([(1, 3), (-4, -2)], axis=1)
    assert_equal(h3d.GetEntries(), h3d_merged.GetEntries())
    assert_equal(h3d.GetSumOfWeights(), h3d_merged.GetSumOfWeights())
    assert_equal(h3d_merged.nbins(1), 6)


def test_rebinning():
    h1d = Hist(100, 0, 1)
    h1d.FillRandom('gaus')
    assert_equal(h1d.rebinned(2).nbins(0), 50)
    assert_equal(h1d.rebinned((2,)).nbins(0), 50)
    assert_equal(h1d.rebinned([0, .5, 1]).nbins(0), 2)

    h3d = Hist3D(10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3d.FillRandom('gaus')
    assert_equal(h3d.rebinned(2).nbins(0), 5)
    new = h3d.rebinned((2, 5, 1))
    assert_equal(new.nbins(0), 5)
    assert_equal(new.nbins(1), 2)
    assert_equal(new.nbins(2), 10)
    new = h3d.rebinned([0, 5, 10], axis=1)
    assert_equal(new.nbins(1), 2)


def test_quantiles():
    h3d = Hist3D(10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3d.FillRandom('gaus')
    h3d.quantiles(2)
    h3d.quantiles(2, axis=1)
    h3d.quantiles([0, .5, 1], axis=2)

    h2d = Hist2D(100, 0, 1, 100, 0, 1)
    h2d.FillRandom(F2('x+y'))
    h2d.quantiles(4, axis=0)
    h2d.quantiles(4, axis=1)


def test_compatibility():
    a = Hist(10, 0, 1)
    b = Hist(10, 0, 1)
    c = Hist(10, 1, 2)
    d = Hist2D(10, 0, 1, 10, 0, 1)

    assert_true(a.compatible(a))
    assert_true(a.compatible(b))
    assert_true(a.compatible(c))
    assert_false(a.compatible(c, check_edges=True))
    assert_false(a.compatible(d))

def test_power():
    h = Hist2D(10, 0, 1, 10, 0, 1)
    h.FillRandom(F2('x+y'))
    p = h.Clone()
    p /= h.Integral()
    pow(h, 2)
    h**2
    h**p
    assert_raises(ValueError, pow, h, Hist2D(20, 0, 1, 10, 0, 1))
    assert_raises(TypeError, pow, h, Hist(10, 0, 1))
    h**=2


def test_integral():
    h = Hist(10, 0, 1)
    h.FillRandom('gaus', 100)
    h[0].value = 2
    h[-1].value = 4
    assert_equal(h.integral(), 100)
    assert_equal(h.integral(overflow=True), 106)
    assert_equal(h.integral(xbin1=1, overflow=True), 104)
    assert_equal(h.integral(xbin2=-2, overflow=True), 102)


def test_integral_error():
    h = Hist(1, 0, 1)
    h.FillRandom('gaus')
    ref_integral, ref_error = h.integral(error=True)

    h1 = Hist(10, 0, 1)
    h1.FillRandom('gaus')
    integral, error = h1.integral(error=True)
    assert_almost_equal(integral, ref_integral)
    assert_almost_equal(error, ref_error)

    h2 = Hist2D(10, 0, 1, 10, 0, 1)
    h2.FillRandom(F2('x+y'))
    integral, error = h2.integral(error=True)
    assert_almost_equal(integral, ref_integral)
    assert_almost_equal(error, ref_error)

    h3 = Hist3D(10, 0, 1, 10, 0, 1, 10, 0, 1)
    h3.FillRandom(F3('x+y+z'))
    integral, error = h3.integral(error=True)
    assert_almost_equal(integral, ref_integral)
    assert_almost_equal(error, ref_error)


def test_poisson_errors():
    h = Hist(20, -3, 3)
    h.FillRandom('gaus')
    g = h.poisson_errors()


def test_overall_efficiency():
    for stat_op in range(0, 8):
        Eff = Efficiency(Hist(20, -3, 3), Hist(20, -3, 3))
        Eff_1bin = Efficiency(Hist(1, -3, 3), Hist(1, -3, 3))
        Eff.SetStatisticOption(stat_op)
        Eff_1bin.SetStatisticOption(stat_op)

        for i in range(1000):
            x = gauss(0, 3.6)
            w = uniform(0, 1)
            passed = w > 0.5
            Eff.Fill(passed, x)
            Eff_1bin.Fill(passed, x)

        assert_almost_equal(Eff.overall_efficiency(overflow=True)[0],
                            Eff_1bin.overall_efficiency(overflow=True)[0])
        assert_almost_equal(Eff.overall_efficiency(overflow=True)[1],
                            Eff_1bin.overall_efficiency(overflow=True)[1])
        assert_almost_equal(Eff.overall_efficiency(overflow=True)[2],
                            Eff_1bin.overall_efficiency(overflow=True)[2])


def test_efficiency():
    # 1D
    eff = Efficiency(Hist(10, 0, 1), Hist(10, 0, 1))
    eff.Fill(False, 0.1)
    eff.Fill(True, 0.8)
    assert_equal(len(eff), len(eff.total))
    if ROOT_VERSION >= ROOTVersion(53417):
        assert eff.graph
    assert eff.painted_graph
    assert_equal(len(list(eff.efficiencies())), 10)
    assert_equal(len(list(eff.efficiencies(overflow=True))), 12)
    assert_equal(len(list(eff.errors())), 10)
    assert_equal(len(list(eff.errors(overflow=True))), 12)
    # 2D
    eff = Efficiency(Hist2D(10, 0, 1, 10, 0, 1), Hist2D(10, 0, 1, 10, 0, 1))
    eff.Fill(False, 0.1)
    eff.Fill(True, 0.8)
    assert_equal(len(eff), len(eff.total))
    if ROOT_VERSION >= ROOTVersion(53417):
        assert eff.histogram
    assert eff.painted_histogram


def test_uniform_binned():
    h = Hist([-5, -4, -1, 2, 5])
    assert_true(not h.uniform())
    h.FillRandom('gaus')
    h_uniform = h.uniform_binned()
    assert_true(h_uniform.uniform())
    assert_equal(h_uniform.entries, h.entries)
    assert_equal(h.integral(), h_uniform.integral())


if __name__ == "__main__":
    import nose
    nose.runmodule()
