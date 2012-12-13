# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Testing for the file module.
"""

from rootpy.io import TemporaryFile, DoesNotExist
from rootpy.plotting import Hist
# http://stackoverflow.com/questions/10716506/where-is-noses-assert-raises-function
from nose.tools import assert_raises  # @UnresolvedImport
import os


def test_tempfile():

    with TemporaryFile():
        Hist(1, 0, 1, name='test').write()


def test_file():

    f = TemporaryFile()
    assert_raises(DoesNotExist, f.Get, 'blah')
    hist = Hist(1, 0, 1, name='test')
    hist.Write()
    hist2 = f.test
    assert hist2.__class__ == hist.__class__
    os.unlink(f.GetName())


if __name__ == "__main__":
    import nose
    nose.runmodule()
