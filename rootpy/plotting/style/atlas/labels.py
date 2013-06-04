# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from ....context import preserve_current_canvas
from ....memory.keepalive import keepalive


def ATLAS_label(x, y, text="Preliminary 20XX", sqrts=8,
                pad=None,
                expfont=73, labelfont=43,
                textsize=20, sep=None):

    if pad is None:
        pad = ROOT.gPad.func()
    with preserve_current_canvas():
        pad.cd()
        l = ROOT.TLatex()
        #l.SetTextAlign(12)
        #l.SetTextSize(tsize)
        l.SetNDC()
        l.SetTextFont(expfont)
        l.SetTextSize(textsize)
        l.SetTextColor(1)
        if sep is None:
            sep = 0.115 * 696 * pad.GetWh() / (472 * pad.GetWw())
        l.DrawLatex(x, y, "ATLAS")
        keepalive(pad, l)
        if text is not None:
            p = ROOT.TLatex()
            p.SetNDC()
            p.SetTextFont(labelfont)
            p.SetTextSize(textsize)
            p.SetTextColor(1)
            if sqrts is not None:
                text = text + " #sqrt{s}=%iTeV" % sqrts
            p.DrawLatex(x + sep, y, text)
            keepalive(pad, p)
        else:
            p = None
        pad.Modified()
        pad.Update()
    return l, p
