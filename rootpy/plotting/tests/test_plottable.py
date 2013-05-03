# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Hist
from nose.tools import assert_equals


def test_plottable_clone():

    a = Hist(10, 0, 1, linecolor='blue', drawstyle='same')

    b = a.Clone(fillstyle='solid')
    assert_equals(b.fillstyle, 'solid')
    assert_equals(b.linecolor, 'blue')
    assert_equals(b.drawstyle, 'same')

    c = a.Clone(color='red')
    assert_equals(c.linecolor, 'red')
    assert_equals(c.fillcolor, 'red')
    assert_equals(c.markercolor, 'red')


if __name__ == "__main__":
    import nose
    nose.runmodule()
