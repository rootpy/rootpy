# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT

from ..core import Object
from .. import defaults, QROOT


class _PadBase(Object):

    def _post_init(self):

        self.members = []

    def Clear(self, *args, **kwargs):

        self.members = []
        self.ROOT_base.Clear(self, *args, **kwargs)

    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

    def cd(self, *args):

        return self.ROOT_base.cd(self, *args)


class Pad(_PadBase, QROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self._post_init()


class Canvas(_PadBase, QROOT.TCanvas):

    def __init__(self,
                 width=None, height=None,
                 x=None, y=None,
                 name=None, title=None):

        # The following line will trigger finalSetup and start the graphics
        # thread if not started already
        style = ROOT.gStyle
        if width is None:
            width = style.GetCanvasDefW()
        if height is None:
            height = style.GetCanvasDefH()
        if x is None:
            x = style.GetCanvasDefX()
        if y is None:
            y = style.GetCanvasDefY()
        Object.__init__(self, name, title, x, y, width, height)
        self._post_init()
