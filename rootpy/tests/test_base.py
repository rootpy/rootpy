# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from rootpy.base import Object
from rootpy.tests.utils import iter_rootpy_classes
from rootpy import asrootpy
from rootpy.io import MemFile
from nose.tools import assert_equal, assert_true


def test_object():

    with MemFile('test', 'recreate'):
        for cls in iter_rootpy_classes():
            # avoid RooStats bugs for now
            if getattr(cls, '_ROOT', object).__name__.startswith('Roo'):
                continue
            if hasattr(cls, 'dynamic_cls'):
                cls = cls.dynamic_cls()
            assert hasattr(cls, '_ROOT'), \
                "rootpy class {0} does not have a _ROOT attribute".format(
                    cls.__name__)
            if issubclass(cls, ROOT.TDirectory):
                continue
            obj = asrootpy(cls._ROOT())

            if isinstance(obj, Object):
                clone = obj.Clone()


if __name__ == "__main__":
    import nose
    nose.runmodule()
