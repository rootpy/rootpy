# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT

from .. import QROOT, asrootpy
from ..core import Object
from .hist import HistStack
from .box import Pave

class Legend(QROOT.TLegend, Pave):

    def __init__(self, nentries,
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

        # ROOT, why are you filling my legend with a grey background by default?
        self.SetFillStyle(0)
        self.SetFillColor(0)

        if textfont is None:
            textfont = ROOT.gStyle.GetLegendFont()
        if textsize is None:
            textsize = ROOT.gStyle.GetTextSize()

        self.SetTextFont(textfont)
        self.SetTextSize(textsize)

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

        If `style` is None, `thing.legendstyle` is used if present otherwise `P`
        """
        if isinstance(thing, HistStack):
            things = thing
        else:
            things = [thing]
        for thing in things:
            if getattr(thing, 'inlegend', True):
                if label is None:
                    label = thing.GetTitle()
                if style is None:
                    style = getattr(thing, 'legendstyle', 'P')
                super(Legend, self).AddEntry(thing, label, style)
    @property
    def primitives(self):
        return asrootpy(self.GetListOfPrimitives())
        
        
