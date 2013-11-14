# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
ATLAS Style, based on a style file from BaBar
"""
from __future__ import absolute_import

from .. import Style

__all__ = [
    'style',
]

def style(name='ATLAS', shape='rect', orientation='landscape'):

    STYLE = Style(name, 'ATLAS Style')

    # turn off borders
    STYLE.SetCanvasBorderMode(0)
    STYLE.SetFrameBorderMode(0)
    STYLE.SetPadBorderMode(0)

    # default canvas size and position
    if shape == 'rect':
        if orientation == 'landscape':
            h = 600
            w = 800
            mtop = 0.05
            mright = 0.04
            mbottom = 0.16
            mleft = 0.16
            xoffset = 1.4
            yoffset = 1.5
        elif orientation == 'portrait':
            h = 800
            w = 600
            mtop = 0.04
            mright = 0.05
            mbottom = 0.12
            mleft = 0.21
            xoffset = 1.4
            yoffset = 2.6
        else:
            raise ValueError("orientation must be 'landscape' or 'portrait'")
    elif shape == 'square':
        h = 600
        w = 600
        mtop = 0.05
        mright = 0.05
        mbottom = 0.16
        mleft = 0.21
        xoffset = 1.4
        yoffset = 2.
    else:
        raise ValueError("shape must be 'square' or 'rect'")

    STYLE.SetCanvasDefH(h)
    STYLE.SetCanvasDefW(w)
    STYLE.SetCanvasDefX(0)
    STYLE.SetCanvasDefY(0)

    # set margin sizes
    STYLE.SetPadTopMargin(mtop)
    STYLE.SetPadRightMargin(mright)
    STYLE.SetPadBottomMargin(mbottom)
    STYLE.SetPadLeftMargin(mleft)

    # set title offsets (for axis label)
    STYLE.SetTitleXOffset(xoffset)
    STYLE.SetTitleYOffset(yoffset)

    # use plain black on white colors
    STYLE.SetFrameFillColor(0)
    STYLE.SetCanvasColor(0)
    STYLE.SetPadColor(0)
    STYLE.SetStatColor(0)

    # don't use white fill color for *all* objects
    #STYLE.SetFillColor(0)

    # NOTE: the following is missing from the official ATLAS style
    STYLE.SetLegendBorderSize(0)
    STYLE.SetLegendFillColor(0)

    # set the paper & margin sizes
    STYLE.SetPaperSize(20,26)

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

    STYLE.SetPalette(1)

    return STYLE
