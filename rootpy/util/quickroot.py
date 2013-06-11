# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Quickly load ROOT symbols without causing slow finalSetup()

The main principle is that appropriate dictionaries need to be loaded.
"""
import ROOT

from .. import log; log = log[__name__]
from ..extern.module_facade import Facade

# Quick's __name__ needs to be the ROOT module for this to be transparent.
# The below is one way of obtaining such a function
Quick = eval("lambda symbol: module._root.LookupRootEntity(symbol)",
             ROOT.__dict__)

Load = Quick("gSystem").Load

# It is not vital to list _all_ symbols in here, just enough that a library
# will be loaded by the time it is needed.
SYMBOLS = dict(
    Hist="TH1 TGraph TGraphAsymmErrors",
    Tree="TCut TTree",
    Gui="TPad TCanvas",
    Graf="TLegend TLine TEllipse",
    Physics="TVector2 TVector3 TLorentzVector TRotation TLorentzRotation",
)

# Mapping of symbols to libraries which need to be loaded
SYMBOLS_TO_LIB = dict(
    (sym, lib) for lib, syms in SYMBOLS.iteritems() for sym in syms.split())

# If you encounter problems with particular symbols, add them to this set.
SLOW = set("".split())

@Facade(__name__, expose_internal=False)
class QuickROOT(object):
    def __getattr__(self, symbol):
        if symbol in SLOW:
            log.warning("Tried to quickly load {0} which is always slow".format(symbol))

        lib = SYMBOLS_TO_LIB.get(symbol, None)
        if lib:
            # Load() doesn't cost anything if the library is already loaded
            libname = "lib{0}".format(lib)
            if Load(libname) == 0:
                log.debug("Loaded {0} (required by {1})".format(
                    libname, symbol))

        return Quick(symbol)

