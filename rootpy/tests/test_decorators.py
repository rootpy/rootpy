# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from rootpy.core import Object
from rootpy.decorators import (method_file_check, method_file_cd,
                               snake_case_methods)
from rootpy.io import TemporaryFile
import rootpy
from nose.tools import assert_true, raises


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
        def write(self): pass

    assert_true(hasattr(B, 'some_method'))
    assert_true(hasattr(B, 'cd'))
    assert_true(hasattr(B, 'long_method_name'))
    assert_true(hasattr(B, 'write'))
    assert_true(hasattr(B, 'other_method'))


class Foo(Object, ROOT.TH1D):

    @method_file_check
    def something(self, foo):
        self.file = rootpy.gDirectory()
        return foo

    @method_file_cd
    def write(self):
        assert_true(self.GetDirectory() == rootpy.gDirectory())


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
