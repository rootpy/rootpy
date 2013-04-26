# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Graph, Graph2D
from nose.tools import assert_equals


def test_init():

    g = Graph(10, name='test')
    assert_equals(len(g), 10)

    g2d = Graph2D(10, name='test2d')


if __name__ == "__main__":
    import nose
    nose.runmodule()
