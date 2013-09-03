# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from ....context import preserve_current_canvas
from ....memory.keepalive import keepalive

__all__ = [
    'ATLAS_label',
]


def ATLAS_label(x, y, text="Preliminary 20XX", sqrts=8,
                pad=None,
                expfont=73, labelfont=43,
                textsize=20, sep=None):

    if pad is None:
        pad = ROOT.gPad.func()
    with preserve_current_canvas():
        pad.cd()
        l = ROOT.TLatex(x, y, "ATLAS")
        #l.SetTextAlign(12)
        #l.SetTextSize(tsize)
        l.SetNDC()
        l.SetTextFont(expfont)
        l.SetTextSize(textsize)
        l.SetTextColor(1)
        l.Draw()
        keepalive(pad, l)
        if sep is None:
            # guess
            sep = 0.115 * 696 * pad.GetWh() / (472 * pad.GetWw())
        if text is not None:
            if sqrts is not None:
                text = text + " #sqrt{{s}}={0:d}TeV".format(sqrts)
            p = ROOT.TLatex(x + sep, y, text)
            p.SetNDC()
            p.SetTextFont(labelfont)
            p.SetTextSize(textsize)
            p.SetTextColor(1)
            p.Draw()
            keepalive(pad, p)
        else:
            p = None
        pad.Modified()
        pad.Update()
    return l, p
