"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT
from ..core import Object
from ..registry import register
from .. import _globals


class PadMixin(object):

    def _post_init(self):

        self.members = []
        _globals.pad = self

    def Clear(self, *args, **kwargs):

        self.members = []
        self.__class__.__bases__[-1].Clear(self, *args, **kwargs)

    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

    def cd(self, *args):

        _globals.pad = self
        return self.__class__.__bases__[-1].cd(self, *args)


@register()
class Pad(Object, PadMixin, ROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self._post_init()


@register()
class Canvas(Object, PadMixin, ROOT.TCanvas):

    def __init__(self, width=800, height=600, xpos=0, ypos=0, name=None, title=None):

        Object.__init__(self, name, title, xpos, ypos, width, height)
        self._post_init()
