# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist, Hist2D, Graph
from nose.plugins.skip import SkipTest


def test_errorbar():
    try:
        from rootpy.plotting import root2matplotlib as rplt
    except ImportError:
        raise SkipTest
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    g = Graph(h)
    rplt.errorbar(g)
    rplt.errorbar(h)


def test_bar():
    try:
        from rootpy.plotting import root2matplotlib as rplt
    except ImportError:
        raise SkipTest
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    rplt.bar(h)


def test_hist():
    try:
        from rootpy.plotting import root2matplotlib as rplt
    except ImportError:
        raise SkipTest
    h = Hist(100, -5, 5)
    h.FillRandom('gaus')
    rplt.hist(h)


def test_hist2d():
    try:
        from rootpy.plotting import root2matplotlib as rplt
        from matplotlib import pyplot
        import numpy as np
    except ImportError:
        raise SkipTest
    if not hasattr(pyplot, 'hist2d'):
        raise SkipTest
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=np.arange(4).reshape(2, 2),
        size=(1E3,)))
    rplt.hist2d(a)


def test_imshow():
    try:
        from rootpy.plotting import root2matplotlib as rplt
        import numpy as np
    except ImportError:
        raise SkipTest
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=np.arange(4).reshape(2, 2),
        size=(1E3,)))
    rplt.imshow(a)


def test_contour():
    try:
        from rootpy.plotting import root2matplotlib as rplt
        import numpy as np
    except ImportError:
        raise SkipTest
    a = Hist2D(100, -3, 3, 100, 0, 6)
    a.fill_array(np.random.multivariate_normal(
        mean=(0, 3),
        cov=np.arange(4).reshape(2, 2),
        size=(1E3,)))
    rplt.contour(a)


if __name__ == "__main__":
    import nose
    nose.runmodule()
