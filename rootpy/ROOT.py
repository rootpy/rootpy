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

import ROOT

from . import asrootpy
from .extern.module_facade import Facade

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
    
    @property
    def gPad(self):
        return self(ROOT.gPad.func())

