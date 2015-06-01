# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy import QROOT
from rootpy.utils.hook import classhook, appendclass, super_overridden
from rootpy.context import invisible_canvas
import rootpy.utils.hook as H

import ROOT


VALUE = 1
ANOTHER = 42


def basicfunc():
    return VALUE, ANOTHER


def wrap():
    a = 1
    def outer(VALUE):
        y = a
        z = 2
        def inner(x):
            return a, x, y, z, VALUE, ANOTHER, nonexist
        return inner
    return outer


def test_inject():
    assert (VALUE, ANOTHER) == (1, 42)
    assert basicfunc() == (VALUE, ANOTHER)

    # Basic injection test
    NEWVALUE = 2
    injected = H.inject_closure_values(basicfunc, VALUE=NEWVALUE)
    assert injected() == (NEWVALUE, ANOTHER)

    # Check that
    try:
        wrap()(1)(2)
    except NameError as e:
        assert "nonexist" in e.args[0]
    else:
        assert False, "expected a NameError"

    global nonexist
    nonexist = 999
    # Test the unmodified version of the function
    correct = wrap()(1)(2)
    del nonexist

    # Test that we can really replace globals
    NEWANOTHER = 43
    newvalue = tuple(a if a != ANOTHER else NEWANOTHER for a in correct)

    hooked = H.inject_closure_values(wrap, ANOTHER=NEWANOTHER, nonexist=999)
    result = hooked()(1)(2)

    assert result == newvalue, ("Closure injection is not working properly")


@classhook(QROOT.TH1)
@super_overridden
class TH1(object):
    def SetTitle(self, *args):
        super(TH1, self).SetTitle(*args)
        return "SUCCESS"

    @classmethod
    def _rootpy_hook_test(cls):
        return "SUCCESS"

    def _rootpy_test_super_draw(self, *args, **kwargs):
        super(TH1, self).Draw(*args, **kwargs)
        return "SUCCESS"


@appendclass(QROOT.TAttLine)
class TAttLine(object):
    @property
    def _rootpy_hook_test_prop(self):
        return "SUCCESS"

    @staticmethod
    def _rootpy_hook_test_static():
        return "SUCCESS"

    @classmethod
    def _rootpy_hook_test_clsmeth(cls):
        return "SUCCESS"

    def _rootpy_hook_test_method(self):
        return "SUCCESS"


def test_hooks():
    h = ROOT.TH1D()

    newtitle = "Hello, world"
    assert h.SetTitle(newtitle) == "SUCCESS"
    assert h.GetTitle() == newtitle

    with invisible_canvas() as c:
        assert c.GetListOfPrimitives().GetSize() == 0
        assert h._rootpy_test_super_draw() == "SUCCESS"
        assert c.GetListOfPrimitives().GetSize() == 1

    assert h._rootpy_hook_test_prop == "SUCCESS"
    assert h._rootpy_hook_test_method() == "SUCCESS"
    assert h._rootpy_hook_test_static() == "SUCCESS"
    assert h._rootpy_hook_test_clsmeth() == "SUCCESS"
