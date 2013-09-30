# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

from rootpy.memory.ownership import GetOwnership

import ROOT as R

def test_getownership():
    o = R.TObject()
    assert GetOwnership(o)
    R.SetOwnership(o, False)
    assert not GetOwnership(o)
