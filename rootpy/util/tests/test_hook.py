from rootpy import QROOT
from rootpy.util.hook import classhook, appendclass

import ROOT

@classhook(QROOT.TH1)
class TH1(object):
    def Draw(self, *args):
        return "SUCCESS"
        res = super(QROOT.TH1, self).Draw(*args)
        return res

@appendclass(QROOT.TAttLine)
class TAttLine(object):
    @property
    def color(self):
        return "SUCCESS"

    @staticmethod
    def static():
        return "SUCCESS"

    @classmethod
    def clsmeth(cls):
        return "SUCCESS"

    def method(self):
        return "SUCCESS"

def test_hooks():
    h = ROOT.TH1D()

    assert h.Draw() == "SUCCESS"
    assert h.color == "SUCCESS"
    assert h.method() == "SUCCESS"
    assert h.static() == "SUCCESS"
    assert h.clsmeth() == "SUCCESS"