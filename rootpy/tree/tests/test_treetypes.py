from rootpy.tree.treetypes import convert
from nose.tools import assert_equal


def test_convert():

    assert_equal(convert('ROOTNAME', 'NUMPY', 'Bool_t'), 'b')


if __name__ == "__main__":
    import nose
    nose.runmodule()
