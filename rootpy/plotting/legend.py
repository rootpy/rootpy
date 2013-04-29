# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT

from .. import QROOT
from ..core import Object
from .core import Plottable
from .hist import HistStack


class Legend(Object, QROOT.TLegend):

    def __init__(self, nentries,
                       pad=None,
                       leftmargin=0.5,
                       topmargin=0.05,
                       rightmargin=0.05,
                       entryheight=0.06):

        height = entryheight * nentries
        if pad is None:
            pad = ROOT.gPad
        super(Legend, self).__init__(
            pad.GetLeftMargin() + leftmargin,
            (1. - pad.GetTopMargin() - topmargin) - height,
            1. - pad.GetRightMargin() - rightmargin,
            ((1. - pad.GetTopMargin()) - topmargin))
        self.pad = pad
        self.UseCurrentStyle()
        self.SetEntrySeparation(0.2)
        self.SetMargin(0.1)
        self.SetFillStyle(0)
        self.SetFillColor(0)
        self.SetTextFont(ROOT.gStyle.GetTextFont())
        self.SetTextSize(ROOT.gStyle.GetTextSize())

    def Height(self):

        return abs(self.GetY2() - self.GetY1())

    def Width(self):

        return abs(self.GetX2() - self.GetX1())

    def Draw(self, *args, **kwargs):

        super(Legend, self).Draw(*args, **kwargs)
        self.UseCurrentStyle()
        self.pad.Modified()
        self.pad.Update()

    def AddEntry(self, thing, legendstyle=None, label=None):
        """
        Add an entry to the legend.

        If legendstyle is None, it will be taken from thing's
        'legendstyle' attribute.

        If label is None, the thing's title will be used as the label.
        """
        if isinstance(thing, HistStack):
            things = thing
        else:
            things = [thing]

        for thing in things:
            if getattr(thing, 'inlegend', True):
                if label is None:
                    label = thing.GetTitle()
                if legendstyle is None:
                    legendstyle = getattr(thing, 'legendstyle', 'P')
                super(Legend, self).AddEntry(thing, label, legendstyle)
        self.pad.Modified()
        self.pad.Update()
