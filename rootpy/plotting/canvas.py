"""
This module implements python classes which inherit from
and extend the functionality of the ROOT canvas classes.
"""

import ROOT

class PadMixin(object):

    def Clear(self, *args, **kwargs):

        self.members = []
        self.__class__.__bases__[-1].Clear(self, *args, **kwargs)
    
    def OwnMembers(self):

        for thing in self.GetListOfPrimitives():
            if thing not in self.members:
                self.members.append(thing)

class Pad(PadMixin, ROOT.TPad):

    def __init__(self, *args, **kwargs):

        ROOT.TPad.__init__(self, *args, **kwargs)
        self.members = []
    
class Canvas(PadMixin, ROOT.TCanvas):

    def __init__(self, *args, **kwargs):

        ROOT.TCanvas.__init__(self, *args, **kwargs)
        self.members = []
 
