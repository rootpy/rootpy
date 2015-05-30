# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Graph, Graph2D, Hist
import tempfile
from random import random
from nose.tools import assert_equal


def test_init():
    g = Graph(10, name='test')
    assert_equal(len(g), 10)
    g2d = Graph2D(10, name='test2d')


def test_init_from_hist():
    h = Hist(100, -10, 10)
    h.FillRandom('gaus')
    g = Graph(h)


def test_init_from_file_1d():
    with tempfile.NamedTemporaryFile() as f:
        for i in range(100):
            f.write('{0:.3f},{1:.3f}\n'.format(
                random(), random()).encode('utf-8'))
        f.flush()
        g = Graph.from_file(f.name, sep=',')
        assert_equal(len(g), 100)


def test_init_from_file_2d():
    with tempfile.NamedTemporaryFile() as f:
        for i in range(100):
            f.write('{0:.3f},{1:.3f},{2:.3f}\n'.format(
                random(), random(), random()).encode('utf-8'))
        f.flush()
        g = Graph2D.from_file(f.name, sep=',')
        assert_equal(len(g), 100)


def test_xerr():
    g = Graph(10)
    list(g.xerr())

    g = Graph(10, type='errors')
    list(g.xerr())

    g = Graph(10, type='asymm')
    list(g.xerr())


def test_divide():
    Graph.divide(Graph(Hist(10, 0, 1).FillRandom('gaus')),
                 Hist(10, 0, 1).FillRandom('gaus'), 'pois')


if __name__ == "__main__":
    import nose
    nose.runmodule()
