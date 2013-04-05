# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

"""
Add the "CMS Preliminary" and \sqrt{s} blurbs to CMS plots.
"""

import ROOT
from ....context import preserve_current_canvas
from ....memory.keepalive import keepalive


def CMS_label(text="Preliminary 2012", sqrts=8, pad=None):
    """ Add a 'CMS Preliminary' style label to the current Pad.

    The blurbs are drawn in the top margin.  The label "CMS " + text is drawn
    in the upper left.  If sqrts is None, it will be omitted.  Otherwise, it
    will be drawn in the upper right.
    """
    if pad is None:
        pad = ROOT.gPad.func()

    with preserve_current_canvas():
        pad.cd()
        l = ROOT.TLatex()
        l.SetTextAlign(12)  # left-middle
        l.SetNDC()
        left_margin = pad.GetLeftMargin()
        top_margin = pad.GetTopMargin()
        ypos = 1 - top_margin / 2.
        # The text is 90% as tall as the margin it lives in.
        l.SetTextSize(0.90 * top_margin)
        l.DrawLatex(left_margin, ypos, "CMS " + text)
        keepalive(pad, l)

        # Draw sqrt(s) label, if desired
        if sqrts:
            p = ROOT.TLatex()
            p.SetTextAlign(32)  # right-middle
            p.SetNDC()
            p.SetTextSize(0.90 * top_margin)
            right_margin = pad.GetRightMargin()
            p.DrawLatex(1 - right_margin, ypos, "#sqrt{s}=%iTeV" % sqrts)
            keepalive(pad, p)
        else:
            p = None
        pad.Modified()
        pad.Update()
    return l, p
