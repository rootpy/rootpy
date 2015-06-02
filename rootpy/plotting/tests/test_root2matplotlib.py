# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist, Hist2D, HistStack, Graph
from nose.plugins.skip import SkipTest
from nose.tools import with_setup


def setup_func():
    try:
        import matplotlib
    except ImportError:
        raise SkipTest("matplotlib is not importable")
    matplotlib.use('Agg')
    from matplotlib import pyplot
    pyplot.ioff()


@with_setup(setup_func)
def test_errorbar():
    from rootpy.plotting import root2matplotlib as rplt
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    g = Graph(h)
    rplt.errorbar(g)
    rplt.errorbar(h)


@with_setup(setup_func)
def test_bar():
    from rootpy.plotting import root2matplotlib as rplt
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    rplt.bar(h)

    # stack
    h1 = h.Clone()
    stack = HistStack([h, h1])
    rplt.bar(stack)
    rplt.bar([h, h1])


@with_setup(setup_func)
def test_hist():
    from rootpy.plotting import root2matplotlib as rplt
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    rplt.hist(h)

    # stack
    h1 = h.Clone()
    stack = HistStack([h, h1])
    rplt.hist(stack)
    rplt.hist([h, h1])


@with_setup(setup_func)
def test_hist2d():
    from rootpy.plotting import root2matplotlib as rplt
    from matplotlib import pyplot
    import numpy as np
    if not hasattr(pyplot, 'hist2d'):
        raise SkipTest("matplotlib is too old")
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=[[1, .5], [.5, 1]],
        size=(1000,)))
    rplt.hist2d(a)


@with_setup(setup_func)
def test_imshow():
    from rootpy.plotting import root2matplotlib as rplt
    import numpy as np
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=[[1, .5], [.5, 1]],
        size=(1000,)))
    rplt.imshow(a)


@with_setup(setup_func)
def test_contour():
    from rootpy.plotting import root2matplotlib as rplt
    import numpy as np
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=[[1, .5], [.5, 1]],
        size=(1000,)))
    rplt.contour(a)


if __name__ == "__main__":
    import nose
    nose.runmodule()
