"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT
from ..core import Object

class PadMixin(object):

    def Clear(self, *args, **kwargs):

        self.members = []
        self.__class__.__bases__[-1].Clear(self, *args, **kwargs)
    
    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

class Pad(Object, PadMixin, ROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self.members = []
    
class Canvas(Object, PadMixin, ROOT.TCanvas):

    def __init__(self, name=None, title=None, xpos=0, ypos=0, width=800, height=600):
         
        Object.__init__(self, name, title, xpos, ypos, width, height)
        self.members = []
 
