# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import numbers

import ROOT

from .. import QROOT, asrootpy
from ..base import Object
from .hist import HistStack
from .box import _Positionable
from ..memory.keepalive import keepalive

__all__ = [
    'Legend',
]


class Legend(_Positionable, Object, QROOT.TLegend):
    _ROOT = QROOT.TLegend

    def __init__(self, entries,
                 pad=None,
                 leftmargin=0.5,
                 topmargin=0.05,
                 rightmargin=0.05,
                 entryheight=0.06,
                 entrysep=0.02,
                 margin=0.3,
                 textfont=None,
                 textsize=None,
                 header=None):
        if pad is None:
            pad = ROOT.gPad.func()
        if not pad:
            raise RuntimeError("create a pad before a legend")

        entries_is_list = False
        if isinstance(entries, numbers.Integral):
            # entries is the expected number of entries that will be included
            # in the legend
            nentries = entries
        else:
            # entries is a list of objects to become entries in the legend
            entries_is_list = True
            nentries = len(entries)

        if header is not None:
            nentries += 1
        height = (entryheight + entrysep) * nentries - entrysep

        super(Legend, self).__init__(
            pad.GetLeftMargin() + leftmargin,
            (1. - pad.GetTopMargin() - topmargin) - height,
            1. - pad.GetRightMargin() - rightmargin,
            ((1. - pad.GetTopMargin()) - topmargin))

        self.SetEntrySeparation(entrysep)
        self.SetMargin(margin)
        if header is not None:
            self.SetHeader(header)

        # ROOT, why are you filling my legend with a
        # grey background by default?
        self.SetFillStyle(0)
        self.SetFillColor(0)

        if textfont is None:
            textfont = ROOT.gStyle.GetLegendFont()
        if textsize is None:
            textsize = ROOT.gStyle.GetTextSize()

        self.SetTextFont(textfont)
        self.SetTextSize(textsize)

        if entries_is_list:
            for thing in entries:
                self.AddEntry(thing)

    def Height(self):
        return abs(self.GetY2() - self.GetY1())

    def Width(self):
        return abs(self.GetX2() - self.GetX1())

    def Draw(self, *args, **kwargs):
        self.UseCurrentStyle()
        super(Legend, self).Draw(*args, **kwargs)

    def AddEntry(self, thing, label=None, style=None):
        """
        Add an entry to the legend.

        If `label` is None, `thing.GetTitle()` will be used as the label.

        If `style` is None, `thing.legendstyle` is used if present,
        otherwise `P`.
        """
        if isinstance(thing, HistStack):
            things = thing
        else:
            things = [thing]
        for thing in things:
            if getattr(thing, 'inlegend', True):
                thing_label = thing.GetTitle() if label is None else label
                thing_style = getattr(thing, 'legendstyle', 'P') if style is None else style
                super(Legend, self).AddEntry(thing, thing_label, thing_style)
                keepalive(self, thing)

    @property
    def primitives(self):
        return asrootpy(self.GetListOfPrimitives())
