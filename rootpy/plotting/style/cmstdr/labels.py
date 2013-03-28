# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

"""
Add the "CMS Preliminary" and \sqrt{s} blurbs to CMS plots.

The blurbs are drawn above the histogram frame.

"""
import ROOT


def CMS_label(text="Preliminary 2012", sqrts=8, pad=None):
    if pad is None:
        pad = ROOT.gPad
    l = ROOT.TLatex()
    l.SetTextAlign(12)  # left-middle
    l.SetNDC()
    left_margin = pad.GetLeftMargin()
    top_margin = pad.GetTopMargin()
    ypos = 1 - top_margin / 2.
    # The text is 90% as tall as the margin it lives in.
    l.SetTextSize(0.90 * top_margin)
    l.DrawLatex(left_margin, ypos, "CMS " + text)

    # Draw sqrt(s) label, if desired
    if sqrts:
        s = ROOT.TLatex()
        s.SetTextAlign(32)  # right-middle
        s.SetNDC()
        s.SetTextSize(0.90 * top_margin)
        right_margin = pad.GetRightMargin()
        s.DrawLatex(1 - right_margin, ypos, "#sqrt{s}=%iTeV" % sqrts)
