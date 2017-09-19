from rootpy.memory.ownership import GetOwnership

import ROOT as R

def test_getownership():
    o = R.TObject()
    assert GetOwnership(o)
    R.SetOwnership(o, False)
    assert not GetOwnership(o)
