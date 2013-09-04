# Copyright 2013 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Add an 'LHCb (Preliminary|Unofficial)' label to plots.
"""
from __future__ import absolute_import

import ROOT

from ....context import preserve_current_canvas
from ....memory.keepalive import keepalive

__all__ = [
    'LHCb_label',
]


def LHCb_label(side="L", status="final", text="", pad=None):
    """Add an 'LHCb (Preliminary|Unofficial)' label to the current pad."""

    if pad is None:
        pad = ROOT.gPad.func()

    with preserve_current_canvas():
        pad.cd()
        if side == "L":
            l = ROOT.TPaveText(pad.GetLeftMargin() + 0.05,
                               0.87 - pad.GetTopMargin(),
                               pad.GetLeftMargin() + 0.30,
                               0.95 - pad.GetTopMargin(),
                               "BRNDC")
        elif side == "R":
            l = ROOT.TPaveText(0.70 - pad.GetRightMargin(),
                               0.75 - pad.GetTopMargin(),
                               0.95 - pad.GetRightMargin(),
                               0.85 - pad.GetTopMargin(),
                               "BRNDC")
        else:
            raise TypeError("Unknown side '{0}'".format(side))

        if status == "final":
            l.AddText("LHCb")
        elif status == "preliminary":
            l.AddText("#splitline{LHCb}{#scale[1.0]{Preliminary}}")
        elif status == "unofficial":
            l.AddText("#splitline{LHCb}{#scale[1.0]{Unofficial}}")
        elif status == "custom":
            l.AddText(text)
        else:
            raise TypeError("Unknown status '{0}'".format(status))

        l.SetFillColor(0)
        l.SetTextAlign(12)
        l.SetBorderSize(0)
        l.Draw()

        keepalive(pad, l)

        pad.Modified()
        pad.Update()

    return l, None
