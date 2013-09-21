# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist, Hist2D, Hist3D, HistStack
from nose.tools import (raises, assert_equal, assert_raises,
                        assert_true, assert_false)


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

@raises(ValueError)
def test_init_edge_order():
    # bin edges must be in ascending order
    Hist2D([10, 2, 30], 10, 1, 5)

@raises(ValueError)
def test_init_edge_repeated():
    # bin edges must not be repeated
    Hist([10, 10, 30])

def test_ravel():

    hist = Hist2D(3, 0, 1, 4, 0, 1)
    for i, bin in enumerate(hist.bins()):
        bin.value = i
        bin.error = i
    rhist = hist.ravel()
    assert_equal(list(rhist.y()), range(12))
    assert_equal(list(rhist.yerrh()), range(12))

def test_uniform():

    hist = Hist(10, 0, 1)
    assert_true(hist.uniform())
    hist = Hist2D(10, 0, 1, [1, 10, 100])
    assert_false(hist.uniform())
    assert_true(hist.uniform(axis=0))

def test_stack():

    stack = HistStack()
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='red'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='blue'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='green'))
    assert_equal(len(stack), 3)

    stack2 = stack.Clone()
    assert_equal(stack2[2].linecolor, 'green')

def test_indexing():

    hist = Hist(10, 0, 1)
    hist.Fill(0.5)
    assert_equal(hist[6].value, 1)
    assert_equal(hist[10].value, 0)
    assert_raises(IndexError, hist.__getitem__, -13)
    assert_raises(IndexError, hist.__getitem__, 12)

def test_slice_assign():
    hist = Hist(10, 0, 1)
    hist[:] = [i for i in xrange(len(hist))]
    assert all([a.value == b for a, b in zip(hist, xrange(len(hist)))])
    clone = hist.Clone()
    # reverse bins
    hist[:] = clone[::-1]
    assert all([a.value == b.value for a, b in zip(hist, clone[::-1])])

def test_slice_assign_bad():
    hist = Hist(10, 0, 1)
    def bad_assign():
        hist[:] = [i for i in xrange(len(hist)+1)]

    assert_raises(RuntimeError, bad_assign)

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
    h2d.FillRandom('gaus')
    h2d.quantiles(4, axis=0)
    h2d.quantiles(4, axis=1)


if __name__ == "__main__":
    import nose
    nose.runmodule()
