"""
Testing for the file module.
"""

from rootpy.io import *
from rootpy.plotting import Hist
from nose.tools import assert_raises
import os


def test_tempfile():

    with TemporaryFile() as f:
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
