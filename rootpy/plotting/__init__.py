# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from .. import log; log = log[__name__]
from .. import IN_IPYTHON_NOTEBOOK
from .. import QROOT
from ..utils.hook import classhook, super_overridden
from ..memory.keepalive import keepalive
from .hist import Hist, Hist1D, Hist2D, Hist3D, Efficiency, HistStack, histogram
from .graph import Graph, Graph1D, Graph2D
from .profile import Profile, Profile1D, Profile2D, Profile3D
from .func import F1, F2, F3
from .legend import Legend
from .canvas import Canvas, Pad
from .style import Style, get_style, set_style

__all__ = [
    'Hist', 'Hist1D', 'Hist2D', 'Hist3D', 'HistStack',
    'Efficiency', 'histogram',
    'Graph', 'Graph1D', 'Graph2D',
    'Profile', 'Profile1D', 'Profile2D', 'Profile3D',
    'F1', 'F2', 'F3',
    'Legend', 'Canvas', 'Pad',
    'Style', 'get_style', 'set_style',
]

if IN_IPYTHON_NOTEBOOK:
    from ..interactive import notebook
    notebook.configure()

@classhook(QROOT.TH1, QROOT.TF1,
           QROOT.THStack,
           QROOT.TGraph, QROOT.TGraph2D,
           QROOT.TBox, QROOT.TText,
           QROOT.TLegend,
           QROOT.TLine, QROOT.TEllipse, QROOT.TArrow)
@super_overridden
class DrawableKeepAlive(object):
    def Draw(self, *args, **kwargs):
        keepalive(ROOT.gPad.func(), self)
        return super(DrawableKeepAlive, self).Draw(*args, **kwargs)
