# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
CMS style from http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/RootMacros/style-CMSTDR.C
"""
from __future__ import absolute_import

from .. import Style

__all__ = [
    'style',
]


def style(name='CMSTDR'):

    STYLE = Style(name, "Style for CMS P-TDR")

    # For the canvas:
    STYLE.SetCanvasBorderMode(0)
    STYLE.SetCanvasColor(0)
    STYLE.SetCanvasDefH(600) # Height of canvas
    STYLE.SetCanvasDefW(600) # Width of canvas
    STYLE.SetCanvasDefX(0)   # Position on screen
    STYLE.SetCanvasDefY(0)

    # For the Pad:
    STYLE.SetPadBorderMode(0)
    # STYLE.SetPadBorderSize(Width_t size = 1)
    STYLE.SetPadColor(0)
    STYLE.SetPadGridX(False)
    STYLE.SetPadGridY(False)
    STYLE.SetGridColor(0)
    STYLE.SetGridStyle(3)
    STYLE.SetGridWidth(1)

    # For the frame:
    STYLE.SetFrameBorderMode(0)
    STYLE.SetFrameBorderSize(1)
    STYLE.SetFrameFillColor(0)
    STYLE.SetFrameFillStyle(0)
    STYLE.SetFrameLineColor(1)
    STYLE.SetFrameLineStyle(1)
    STYLE.SetFrameLineWidth(1)

    # For the histo:
    # STYLE.SetHistFillColor(1)
    # STYLE.SetHistFillStyle(0)
    STYLE.SetHistLineColor(1)
    STYLE.SetHistLineStyle(0)
    STYLE.SetHistLineWidth(1)
    # STYLE.SetLegoInnerR(Float_t rad = 0.5)
    # STYLE.SetNumberContours(Int_t number = 20)

    STYLE.SetEndErrorSize(2)
    #STYLE.SetErrorMarker(20)  # Seems to give an error
    STYLE.SetErrorX(0.)

    STYLE.SetMarkerStyle(20)

    #For the fit/function:
    STYLE.SetOptFit(1)
    STYLE.SetFitFormat("5.4g")
    STYLE.SetFuncColor(2)
    STYLE.SetFuncStyle(1)
    STYLE.SetFuncWidth(1)

    #For the date:
    STYLE.SetOptDate(0)
    # STYLE.SetDateX(Float_t x = 0.01)
    # STYLE.SetDateY(Float_t y = 0.01)

    # For the statistics box:
    STYLE.SetOptFile(0)
    STYLE.SetOptStat(0) # To display the mean and RMS:   SetOptStat("mr")
    STYLE.SetStatColor(0)
    STYLE.SetStatFont(42)
    STYLE.SetStatFontSize(0.025)
    STYLE.SetStatTextColor(1)
    STYLE.SetStatFormat("6.4g")
    STYLE.SetStatBorderSize(1)
    STYLE.SetStatH(0.1)
    STYLE.SetStatW(0.15)
    # STYLE.SetStatStyle(Style_t style = 1001)
    # STYLE.SetStatX(Float_t x = 0)
    # STYLE.SetStatY(Float_t y = 0)

    # Margins:
    STYLE.SetPadTopMargin(0.05)
    STYLE.SetPadBottomMargin(0.13)
    STYLE.SetPadLeftMargin(0.16)
    STYLE.SetPadRightMargin(0.02)

    # For the Global title:
    STYLE.SetOptTitle(0)    # 0=No Title
    STYLE.SetTitleFont(42)
    STYLE.SetTitleColor(1)
    STYLE.SetTitleTextColor(1)
    STYLE.SetTitleFillColor(10)
    STYLE.SetTitleFontSize(0.05)
    # STYLE.SetTitleH(0) # Set the height of the title box
    # STYLE.SetTitleW(0) # Set the width of the title box
    # STYLE.SetTitleX(0) # Set the position of the title box
    # STYLE.SetTitleY(0.985) # Set the position of the title box
    # STYLE.SetTitleStyle(Style_t style = 1001)
    # STYLE.SetTitleBorderSize(2)

    # For the axis titles:
    STYLE.SetTitleColor(1, "XYZ")
    STYLE.SetTitleFont(42, "XYZ")
    STYLE.SetTitleSize(0.05, "XYZ")
    # STYLE.SetTitleXSize(Float_t size = 0.02) # Another way to set the size?
    # STYLE.SetTitleYSize(Float_t size = 0.02)
    STYLE.SetTitleXOffset(1.0)
    STYLE.SetTitleYOffset(1.35)
    # STYLE.SetTitleOffset(1.1, "Y") # Another way to set the Offset

    # For the axis labels:
    STYLE.SetLabelColor(1, "XYZ")
    STYLE.SetLabelFont(42, "XYZ")
    STYLE.SetLabelOffset(0.007, "XYZ")
    STYLE.SetLabelSize(0.04, "XYZ")

    # For the axis:
    STYLE.SetAxisColor(1, "XYZ")
    STYLE.SetStripDecimals(True)
    STYLE.SetTickLength(0.03, "XYZ")
    STYLE.SetNdivisions(510, "XYZ")
    STYLE.SetPadTickX(1)  # 0=Text labels (and tics) only on bottom, 1=Text labels on top and bottom
    STYLE.SetPadTickY(1)

    # Change for log plots:
    STYLE.SetOptLogx(0)
    STYLE.SetOptLogy(0)
    STYLE.SetOptLogz(0)

    # Postscript options:
    STYLE.SetPaperSize(20.,20.)
    # STYLE.SetLineScalePS(Float_t scale = 3)
    # STYLE.SetLineStyleString(Int_t i, const char* text)
    # STYLE.SetHeaderPS(const char* header)
    # STYLE.SetTitlePS(const char* pstitle)

    # STYLE.SetBarOffset(Float_t baroff = 0.5)
    # STYLE.SetBarWidth(Float_t barwidth = 0.5)
    # STYLE.SetPaintTextFormat(const char* format = "g")
    # STYLE.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
    # STYLE.SetTimeOffset(Double_t toffset)
    # STYLE.SetHistMinimumZero(True)

    STYLE.SetPalette(1)

    return STYLE
