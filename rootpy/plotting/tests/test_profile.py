# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.plotting import Profile, Profile2D, Profile3D
from nose.tools import raises


def test_init():
    # constructor arguments are repetitions of #bins, left bound, right bound.
    p2d = Profile2D(10, 0, 1, 50, -40, 10, name='2d profile')
    p3d = Profile3D(3, -1, 4, 10, -1000, -200, 2, 0, 1, name='3d profile')

    # variable-width bins may be created by passing the bin edges directly:
    p1d_variable = Profile([1, 4, 10, 100])
    p2d_variable = Profile2D([2, 4, 7, 100, 200], [-100, -50, 0, 10, 20])
    p3d_variable = Profile3D([1, 3, 10], [20, 50, 100], [-10, -5, 10, 20])

    # variable-width and constant-width bins can be mixed:
    p2d_mixed = Profile2D([2, 10, 30], 10, 1, 5)

def test_init_profiled_edges():
    # specifying the profiled axis bounds is optional
    p1d_variable = Profile([1, 4, 10, 100], 0, 1)
    # ROOTBUG: missing constructor:
    #p2d_variable = Profile2D([2, 4, 7, 100, 200], [-100, -50, 0, 10, 20], 0, 1)
    #p3d_variable = Profile3D([1, 3, 10], [20, 50, 100], [-10, -5, 10, 20], 0, 1)

def test_init_option():
    # specifying profile options is optional
    p1d_variable = Profile([1, 4, 10, 100], option='s')
    p2d_variable = Profile2D([2, 4, 7, 100, 200], [-100, -50, 0, 10, 20],
            option='s')
    p3d_variable = Profile3D([1, 3, 10], [20, 50, 100], [-10, -5, 10, 20],
            option='s')
    p1d_variable = Profile([1, 4, 10, 100], 0, 1, option='s')
    # ROOTBUG: missing constructor:
    #p2d_variable = Profile2D([2, 4, 7, 100, 200], [-100, -50, 0, 10, 20], 0, 1,
    #        option='s')
    p3d_variable = Profile3D([1, 3, 10], [20, 50, 100], [-10, -5, 10, 20],
            option='s')

@raises(ValueError)
def test_init_edge_order():
    # bin edges must be in ascending order
    Profile2D([10, 2, 30], 10, 1, 5)

@raises(ValueError)
def test_init_edge_repeated():
    # bin edges must not be repeated
    Profile([10, 10, 30])

@raises(ValueError)
def test_init_profiled_edge_order():
    # profiled axis edges must be in ascending order
    Profile2D([10, 2, 30], 10, 1, 5, 3, 1)

@raises(ValueError)
def test_init_profiled_edge_repeated():
    # bin edges must not be repeated
    Profile([1, 10, 30], 1, 1)


if __name__ == "__main__":
    import nose
    nose.runmodule()
