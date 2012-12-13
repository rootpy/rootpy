# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT


def ATLAS_label(x, y, text=None, color=1, pad=None):

    l = ROOT.TLatex() #l.SetTextAlign(12); l.SetTextSize(tsize)
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    if pad is None:
        pad = ROOT.pPad
    delx = 0.115 * 696 * pad.GetWh() / (472 * pad.GetWw())
    l.DrawLatex(x, y, "ATLAS")
    if text is not None:
        p = ROOT.TLatex()
        p.SetNDC()
        p.SetTextFont(42)
        p.SetTextColor(color)
        p.DrawLatex(x + delx, y, text)
        #p.DrawLatex(x,y,"#sqrt{s}=900GeV")

def ATLAS_version(version, x=0.88, y=0.975, color=1, pad=None):

    l = ROOT.TLatex()
    l.SetTextAlign(22)
    l.SetTextSize(0.04)
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    l.DrawLatex(x, y, "Version %s" % version)
