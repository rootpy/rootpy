# Copyright 2014 the rootpy developers

from rootpy.tree import Cut
from nose.tools import assert_equal


def test_safe():
    assert_equal(Cut("var**2").safe(), "var_pow_2")
    assert_equal(Cut("var*2").safe(), "var_mul_2")
    assert_equal(Cut("2*var**2").safe(), "2_mul_var_pow_2")
    assert_equal(Cut("-4+1<fabs(var)*1e-3<4*10").safe(), "L-4+1_lt_fabsLvarR_mul_1e-3R_and_LfabsLvarR_mul_1e-3_lt_4_mul_10R")

if __name__ == "__main__":
    import nose
    nose.runmodule()
