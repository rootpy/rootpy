# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
:py:mod:`rootpy.ROOT`
====================

Mimick PyROOT's interface, intended to be pluggable replacement for ordinary
PyROOT imports.

If you find a case where it is not, please report an issue to the rootpy
developers.

Plain old root can be accessed through the ``R`` property.

Example use:

.. sourcecode:: ipython

    In [1]: import rootpy.ROOT as ROOT

    In [2]: ROOT.TFile
    Out[2]: rootpy.io.file.File

    In [3]: ROOT.R.TFile
    Out[3]: ROOT.TFile

    In [4]: from rootpy.ROOT import TFile

    In [5]: TFile
    Out[5]: rootpy.io.file.File
"""
from __future__ import absolute_import
from copy import copy

import ROOT

from . import asrootpy, ROOT_VERSION
from .extern.module_facade import Facade


def proxy_global(name, no_expand_macro=False):
    """
    Used to automatically asrootpy ROOT's thread local variables
    """
    if no_expand_macro:
        # handle older ROOT versions without _ExpandMacroFunction wrapping

        @property
        def gSomething_no_func(self):
            glob = self(getattr(ROOT, name))
            # create a fake func() that just returns self
            def func():
                return glob
            glob.func = func
            return glob
        return gSomething_no_func

    @property
    def gSomething(self):

        glob = getattr(ROOT, name)

        orig_func = glob.func

        def asrootpy_izing_func():
            return self(orig_func())

        new_glob = copy(glob)
        new_glob.func = asrootpy_izing_func

        # Memoize
        setattr(type(self), name, new_glob)

        return new_glob
    return gSomething


@Facade(__name__, expose_internal=False)
class Module(object):

    def __call__(self, arg):
        return asrootpy(arg, warn=False)

    def __getattr__(self, what):

        try:
            # Try the version with a T infront first, so that we can raise the
            # correct exception if it doesn't work.
            # (This assumes there are no cases where X and TX are both valid
            #  ROOT classes)
            result = getattr(ROOT, "T" + what)
        except AttributeError:
            try:
                result = getattr(ROOT, what)
            except AttributeError:
                raise

        result = self(result)
        setattr(self, what, result)
        return result

    @property
    def R(self):
        return ROOT

    gPad = proxy_global("gPad")
    gVirtualX = proxy_global("gVirtualX")

    if ROOT_VERSION < (5, 32, 0):
        # handle versions of ROOT older than 5.32.00
        gDirectory = proxy_global("gDirectory", no_expand_macro=True)
        gFile = proxy_global("gFile", no_expand_macro=True)
        gInterpreter = proxy_global("gInterpreter", no_expand_macro=True)
    else:
        gDirectory = proxy_global("gDirectory")
        gFile = proxy_global("gFile")
        gInterpreter = proxy_global("gInterpreter")
