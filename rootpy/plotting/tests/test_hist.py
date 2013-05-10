# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist, Hist2D, Hist3D, HistStack
from nose.tools import raises, assert_equals, assert_raises


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

    content = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]

    errors = [
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
    ]

    hist = Hist2D(3, 0, 1, 4, 0, 1)
    for i in xrange(3):
        for j in xrange(4):
            hist[i, j] = content[i][j]
            hist.SetBinError(i + 1, j + 1, errors[i][j])

    rhist = hist.ravel()
    assert_equals(list(rhist), range(1, 13))
    assert_equals(list(rhist.yerrh()), range(2, 14))

def test_stack():

    stack = HistStack()
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='red'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='blue'))
    stack.Add(Hist(10, 0, 1, fillstyle='solid', color='green'))
    assert_equals(len(stack), 3)

    stack2 = stack.Clone()
    assert_equals(stack2[2].linecolor, 'green')

def test_indexing():

    hist = Hist(10, 0, 1)
    hist.Fill(0.5)
    assert_equals(hist[5], 1)
    assert_equals(hist[9], 0)
    assert_raises(IndexError, hist.__getitem__, -1)
    assert_raises(IndexError, hist.__getitem__, 10)


if __name__ == "__main__":
    import nose
    nose.runmodule()
