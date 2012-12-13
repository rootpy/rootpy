# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.tree.categories import Categories
from nose.tools import assert_raises

GOOD = [
    '{a|1,2,3}',
    '{var1|1,2,3}x{var2|-10,10.3,100*}x{var3|100}',
]

BAD = [
    '{1,2,3}',
    '{a|3,2,1}',
    '{var1|1,2*,3}',
]

def test_from_string():

    for s in GOOD:
        Categories.from_string(s)
    for s in BAD:
        assert_raises(SyntaxError, Categories.from_string, s)

def test_len():

    c = Categories.from_string('{a|1,2,3}')
    assert len(c) == 4
    assert len(c) == len(list(c))

    c = Categories.from_string('{a|1,2,3}x{b|4,5,6}')
    assert len(c) == 16

    c = Categories.from_string('{a|1,2,3}x{b|4,5,6*}')
    assert len(c) == 12

    c = Categories.from_string('{a|1,2,3}x{b|*4,5,6*}')
    assert len(c) == 8

    c = Categories.from_string('{a|1,2,3*}x{b|*4,5,6*}')
    assert len(c) == 6

    c = Categories.from_string('{a|*1,2,3*}x{b|*4,5,6*}')
    assert len(c) == 4


if __name__ == "__main__":
    import nose
    nose.runmodule()
