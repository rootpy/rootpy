# Copyright 2014 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.tree import Cut
from nose.tools import assert_equal


def test_safe():
    assert_equal(Cut("var**2").safe(), "var_pow_2")
    assert_equal(Cut("var*2").safe(), "var_mul_2")
    assert_equal(Cut("2*var**2").safe(), "2_mul_var_pow_2")

if __name__ == "__main__":
    import nose
    nose.runmodule()
