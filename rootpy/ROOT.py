# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
:py:mod:`rootpy.ROOT`
=====================

This module is intended to be a drop-in replacement for ordinary
PyROOT imports by mimicking PyROOT's interface. If you find a case where it is
not, please report an issue to the rootpy developers.

Both ROOT and rootpy classes can be accessed in a harmonized way through this
module. This means you can take advantage of rootpy classes automatically by
replacing ``import ROOT`` with ``import rootpy.ROOT as ROOT`` or
``from rootpy import ROOT`` in your code, while maintaining backward
compatibility with existing use of ROOT's classes.

ROOT classes are automatically "asrootpy'd" *after* the constructor in ROOT has
been called:

.. sourcecode:: python

   >>> import rootpy.ROOT as ROOT
   >>> h = ROOT.TH1F('name', 'title', 10, 0, 1)
   >>> h
   Hist('name')
   >>> h.TYPE
   'F'

Also access rootpy classes under this same module without needing to remember
where to import them from in rootpy:

.. sourcecode:: python

   >>> import rootpy.ROOT as ROOT
   >>> h = ROOT.Hist(10, 0, 1, name='name', type='F')
   >>> h
   Hist('name')
   >>> h.TYPE
   'F'

Plain old ROOT can still be accessed through the ``R`` property:

.. sourcecode:: python

   >>> from rootpy import ROOT
   >>> ROOT.R.TFile
   <class 'ROOT.TFile'>

"""
from __future__ import absolute_import

from copy import copy

import ROOT

from . import asrootpy, lookup_rootpy, ROOT_VERSION
from . import QROOT, stl
from .utils.module_facade import Facade

__all__ = []


def proxy_global(name, no_expand_macro=False, fname='func', args=()):
    """
    Used to automatically asrootpy ROOT's thread local variables
    """
    if no_expand_macro:  # pragma: no cover
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
        obj_func = getattr(getattr(ROOT, name), fname)
        try:
            obj = obj_func(*args)
        except ReferenceError:  # null pointer
            return None
        # asrootpy
        return self(obj)

    return gSomething


@Facade(__name__, expose_internal=False)
class Module(object):

    __version__ = ROOT_VERSION

    def __call__(self, arg, after_init=False):
        return asrootpy(arg, warn=False, after_init=after_init)

    def __getattr__(self, what):
        try:
            # check ROOT
            result = self(getattr(ROOT, what), after_init=True)
        except AttributeError:
            # check rootpy
            result = lookup_rootpy(what)
            if result is None:
                raise AttributeError(
                    'ROOT does not have the attribute `{0}` '
                    'and rootpy does not contain the class `{0}`'.format(what))
            return result

        try:
            # Memoize
            setattr(self, what, result)
        except AttributeError:
            # Oops... Oh well. I tried.
            pass

        return result

    @property
    def R(self):
        return ROOT

    gPad = proxy_global("gPad",
        fname='GetPad' if ROOT_VERSION >= (6, 9, 2) else 'func',
        args=(0,) if ROOT_VERSION >= (6, 9, 2) else ())
    gVirtualX = proxy_global("gVirtualX")

    if ROOT_VERSION < (5, 32, 0):  # pragma: no cover
        gDirectory = proxy_global("gDirectory", no_expand_macro=True)
        gFile = proxy_global("gFile", no_expand_macro=True)
        gInterpreter = proxy_global("gInterpreter", no_expand_macro=True)
    else:
        gDirectory = proxy_global("gDirectory",
            fname='CurrentDirectory' if ROOT_VERSION >= (6, 9, 2) else 'func')
        gFile = proxy_global("gFile",
            fname='CurrentFile' if ROOT_VERSION >= (6, 9, 2) else 'func')
        gInterpreter = proxy_global("gInterpreter",
            no_expand_macro=ROOT_VERSION >= (6, 9, 2))

    # use the smart template STL types from rootpy.stl instead
    for t in QROOT.std.stlclasses:
        locals()[t] = getattr(stl, t)
    del t
