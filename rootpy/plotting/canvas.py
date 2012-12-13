# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT

from ..core import Object
from .. import rootpy_globals as _globals
from .. import defaults, QROOT


class _PadBase(Object):

    def _post_init(self):

        self.members = []
        _globals.pad = self

    def Clear(self, *args, **kwargs):

        self.members = []
        self.ROOT_base.Clear(self, *args, **kwargs)

    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

    def cd(self, *args):

        _globals.pad = self
        return self.ROOT_base.cd(self, *args)

class Pad(_PadBase, QROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self._post_init()

class Canvas(_PadBase, QROOT.TCanvas):

    def __init__(self,
                 width=defaults.CANVAS_WIDTH,
                 height=defaults.CANVAS_HEIGHT,
                 xpos=0, ypos=0, name=None, title=None):

        # trigger finalSetup and start graphics thread if not started already
        ROOT.kTRUE
        Object.__init__(self, name, title, xpos, ypos, width, height)
        self._post_init()
