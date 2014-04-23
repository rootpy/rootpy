# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting.utils import _limits_helper
from nose.tools import assert_equal, assert_raises


def test_limits():
    assert_equal(_limits_helper(0, 1, 0, 0), (0, 1))
    assert_equal(_limits_helper(1, 1, 0, 0, snap=True), (0, 1))
    assert_equal(_limits_helper(-2, -1, 0, 0, snap=True), (-2, 0))
    assert_equal(_limits_helper(-1, 1, .1, .1, snap=True), (-1.25, 1.25))


if __name__ == "__main__":
    import nose
    nose.runmodule()
