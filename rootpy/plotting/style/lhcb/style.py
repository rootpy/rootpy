# Copyright 2013 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
LHCb style from lhcbStyle.C
"""
from __future__ import absolute_import

from .. import Style

__all__ = [
    'style',
]


def style(name='LHCb'):

    STYLE = Style(name, 'LHCb Style')

    font = 132
    line_width = 2
    text_size = 0.06

    # Default canvas size and position.
    STYLE.SetCanvasDefH(600)
    STYLE.SetCanvasDefW(800)
    STYLE.SetCanvasDefX(0)
    STYLE.SetCanvasDefY(0)

    # Colours.
    STYLE.SetCanvasBorderMode(0)
    STYLE.SetCanvasColor(0)
    STYLE.SetFillColor(1)
    STYLE.SetFillStyle(1001)
    STYLE.SetFrameBorderMode(0)
    STYLE.SetFrameFillColor(0)
    STYLE.SetLegendBorderSize(0)
    STYLE.SetPadBorderMode(0)
    STYLE.SetPadColor(0)
    STYLE.SetPalette(1)
    STYLE.SetStatColor(0)

    # Paper and margin sizes.
    STYLE.SetPadBottomMargin(0.16)
    STYLE.SetPadLeftMargin(0.14)
    STYLE.SetPadRightMargin(0.05)
    STYLE.SetPadTopMargin(0.05)
    STYLE.SetPaperSize(20, 26)

    # Font.
    STYLE.SetLabelFont(font, "x")
    STYLE.SetLabelFont(font, "y")
    STYLE.SetLabelFont(font, "z")
    STYLE.SetLabelSize(text_size, "x")
    STYLE.SetLabelSize(text_size, "y")
    STYLE.SetLabelSize(text_size, "z")
    STYLE.SetTextFont(font)
    STYLE.SetTextSize(text_size)
    STYLE.SetTitleFont(font)
    STYLE.SetTitleFont(font, "x")
    STYLE.SetTitleFont(font, "y")
    STYLE.SetTitleFont(font, "z")
    STYLE.SetTitleSize(1.2*text_size, "x")
    STYLE.SetTitleSize(1.2*text_size, "y")
    STYLE.SetTitleSize(1.2*text_size, "z")

    # Lines and markers.
    STYLE.SetFrameLineWidth(line_width)
    STYLE.SetFuncWidth(line_width)
    STYLE.SetGridWidth(line_width)
    STYLE.SetHistLineWidth(line_width)
    STYLE.SetLineStyleString(2, "[12 12]")
    STYLE.SetLineWidth(line_width)
    STYLE.SetMarkerSize(1.0)
    STYLE.SetMarkerStyle(20)

    # Label offsets.
    STYLE.SetLabelOffset(0.010, "X")
    STYLE.SetLabelOffset(0.010, "Y")

    # Decorations.
    STYLE.SetOptFit(0)
    STYLE.SetOptStat(0)
    STYLE.SetOptTitle(0)
    STYLE.SetStatFormat("6.3g")

    # Titles.
    STYLE.SetTitleBorderSize(0)
    STYLE.SetTitleFillColor(0)
    STYLE.SetTitleFont(font, "title")
    STYLE.SetTitleH(0.05)
    STYLE.SetTitleOffset(0.95, "X")
    STYLE.SetTitleOffset(0.95, "Y")
    STYLE.SetTitleOffset(1.2, "Z")
    STYLE.SetTitleStyle(0)
    STYLE.SetTitleW(1.0)
    STYLE.SetTitleX(0.0)
    STYLE.SetTitleY(1.0)

    # Statistics box.
    STYLE.SetStatBorderSize(0)
    STYLE.SetStatFont(font)
    STYLE.SetStatFontSize(0.05)
    STYLE.SetStatH(0.15)
    STYLE.SetStatW(0.25)
    STYLE.SetStatX(0.9)
    STYLE.SetStatY(0.9)

    # Tick marks.
    STYLE.SetPadTickX(1)
    STYLE.SetPadTickY(1)

    # Divisions: only 5 in x to avoid label overlaps.
    STYLE.SetNdivisions(505, "x")
    STYLE.SetNdivisions(510, "y")

    return STYLE
