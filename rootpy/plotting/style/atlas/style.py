# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
ATLAS Style, based on a style file from BaBar
"""
from .. import Style

STYLE = Style("ATLAS", "Atlas style")

# use plain black on white colors
icol = 0 # WHITE
STYLE.SetFrameBorderMode(icol)
STYLE.SetFrameFillColor(icol)
STYLE.SetCanvasBorderMode(icol)
STYLE.SetCanvasColor(icol)
STYLE.SetPadBorderMode(icol)
STYLE.SetPadColor(icol)
STYLE.SetStatColor(icol)
# don't use: white fill color for *all* objects
#STYLE.SetFillColor(icol)
# NOTE: the following is missing from the official ATLAS style
STYLE.SetLegendBorderSize(0)
STYLE.SetLegendFillColor(0)

# set the paper & margin sizes
STYLE.SetPaperSize(20,26)

# set margin sizes
STYLE.SetPadTopMargin(0.05)
STYLE.SetPadRightMargin(0.05)
STYLE.SetPadBottomMargin(0.16)
STYLE.SetPadLeftMargin(0.16)

# set title offsets (for axis label)
STYLE.SetTitleXOffset(1.4)
STYLE.SetTitleYOffset(1.4)

# use large fonts
#font = 72 # Helvetica italics
# NOTE: the official ATLAS style uses 42 here but it is preferred to specify the
# font size in pixels, independent of the canvas size
font = 43 # Helvetica
tsize = 30
STYLE.SetTextFont(font)
STYLE.SetLegendFont(font)

STYLE.SetTextSize(tsize)
STYLE.SetLabelFont(font, "x")
STYLE.SetTitleFont(font, "x")
STYLE.SetLabelFont(font, "y")
STYLE.SetTitleFont(font, "y")
STYLE.SetLabelFont(font, "z")
STYLE.SetTitleFont(font, "z")

STYLE.SetLabelSize(tsize, "x")
STYLE.SetTitleSize(tsize, "x")
STYLE.SetLabelSize(tsize, "y")
STYLE.SetTitleSize(tsize, "y")
STYLE.SetLabelSize(tsize, "z")
STYLE.SetTitleSize(tsize, "z")

# use bold lines and markers
STYLE.SetMarkerStyle(20)
STYLE.SetMarkerSize(1.2)
STYLE.SetHistLineWidth(2)
STYLE.SetLineStyleString(2, "[12 12]") # postscript dashes

# get rid of X error bars
#STYLE.SetErrorX(0.001)
# get rid of error bar caps
STYLE.SetEndErrorSize(0.)

# do not display any of the standard histogram decorations
STYLE.SetOptTitle(0)
#STYLE.SetOptStat(1111)
STYLE.SetOptStat(0)
#STYLE.SetOptFit(1111)
STYLE.SetOptFit(0)

# put tick marks on top and RHS of plots
STYLE.SetPadTickX(1)
STYLE.SetPadTickY(1)
