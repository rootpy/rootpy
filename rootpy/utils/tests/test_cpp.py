# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import print_function
import sys
from ROOT import MethodProxy
import inspect
from rootpy.utils.cpp import CPPGrammar
from rootpy.utils.extras import iter_ROOT_classes
from nose.plugins.attrib import attr


@attr('slow')
def test_cpp():

    i = 0
    num_methods = 0

    for cls in iter_ROOT_classes():
        members = inspect.getmembers(cls)
        # filter out those starting with "_" or "operator "
        # and non-method members
        # also split overloaded methods
        methods = {}
        for name, func in members:
            if name.startswith('_') or name.startswith('operator'):
                continue
            if not isinstance(func, MethodProxy):
                continue
            methods[name] = (func, func.func_doc.split('\n'))

        for name, (func, sigs) in methods.items():
            for sig in sigs:
                num_methods += 1
                if CPPGrammar.parse_method(sig, silent=False):
                    i += 1
            print("{0} / {1}".format(i, num_methods), end='\r')
            sys.stdout.flush()
    print("{0} / {1}".format(i, num_methods))

if __name__ == "__main__":
    import nose
    nose.runmodule()
