# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy import ROOT
from rootpy.base import Object
from rootpy.decorators import (method_file_check, method_file_cd,
                               snake_case_methods)
from rootpy.io import TemporaryFile
import rootpy
from nose.tools import assert_equal, assert_true, raises


def test_snake_case_methods():

    class A(object):
        def SomeMethod(self): pass
        def some_method(self): pass
        def OtherMethod(self): pass
        def Write(self): pass
        def Cd(self): pass
        def cd(self): pass
        def LongMethodName(self): pass

    @snake_case_methods
    class B(A):
        _ROOT = A
        def write(self): pass

    assert_true(hasattr(B, 'some_method'))
    assert_true(hasattr(B, 'cd'))
    assert_true(hasattr(B, 'long_method_name'))
    assert_true(hasattr(B, 'write'))
    assert_true(hasattr(B, 'other_method'))


def test_snake_case_methods_descriptor():

    def f(_): pass

    class A(object):
        Prop = property(f)
        Sm = staticmethod(f)
        Cm = classmethod(f)
        M = f

    class B(A):
        cm = A.__dict__["Cm"]
        m = A.__dict__["M"]
        prop = A.__dict__["Prop"]
        sm = A.__dict__["Sm"]

    @snake_case_methods
    class snakeB(A):
        _ROOT = A

    # Ensure that no accidental descriptor dereferences happened inside
    # `snake_case_methods`. This is checked by making sure that the types
    # are the same between B and snakeB.

    for member in dir(snakeB):
        if member.startswith("_"): continue
        assert_equal(type(getattr(B, member)), type(getattr(snakeB, member)))


class Foo(Object, ROOT.R.TH1D):

    @method_file_check
    def something(self, foo):
        self.file = ROOT.gDirectory.func()
        return foo

    @method_file_cd
    def write(self):
        assert_true(self.GetDirectory() == ROOT.gDirectory.func())


def test_method_file_check_good():

    foo = Foo()
    with TemporaryFile():
        foo.something(42)


@raises(RuntimeError)
def test_method_file_check_bad():

    foo = Foo()
    foo.something(42)


def test_method_file_cd():

    file1 = TemporaryFile()
    foo = Foo()
    foo.SetDirectory(file1)
    file2 = TemporaryFile()
    foo.write()


if __name__ == "__main__":
    import nose
    nose.runmodule()
