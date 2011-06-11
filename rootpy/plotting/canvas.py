"""
This module implements python classes which inherit from
and extend the functionality of the ROOT histogram and graph classes.

These histogram classes may be used within other plotting frameworks like
matplotlib while maintaining full compatibility with ROOT.
"""

from operator import add, sub
from rootpy.objectproxy import ObjectProxy
from rootpy.core import *
from rootpy.registry import *
import math
import ROOT

"""
try:
    from numpy import array
except:
"""
from array import array

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
 
