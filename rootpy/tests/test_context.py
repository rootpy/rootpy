from rootpy import ROOT
from rootpy.context import *
from rootpy.plotting import Canvas
from rootpy.io import TemporaryFile
from nose.tools import assert_equal, assert_true, raises


def test_preserve_current_directory():
    with preserve_current_directory():
        f = TemporaryFile()
        f.Close()

    f = TemporaryFile()
    with preserve_current_directory():
        with TemporaryFile() as g:
            assert_true(g.GetName() == ROOT.gDirectory.GetName())
    assert_true(f.GetName() == ROOT.gDirectory.GetName())


def test_preserve_current_canvas():
    with preserve_current_canvas():
        c = Canvas()

    c = Canvas(name='c')
    with preserve_current_canvas():
        d = Canvas(name='d')
        assert_true(d.GetName() == ROOT.gPad.GetName())
    assert_true(c.GetName() == ROOT.gPad.GetName())


if __name__ == "__main__":
    import nose
    nose.runmodule()
