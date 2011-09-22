from ROOT import TLorentzVector
from copy import copy

class FourVector(TLorentzVector):

    def __init__(self, *args, **kwargs):

        TLorentzVector.__init__(self, *args, **kwargs)

    def __add__(self, other):
        
        clone = copy(self)
        if other:
            return TLorentzVector.__add__(clone, other)
        return clone
    
    def __iadd__(self, other):
        
        TLorentzVector.__add__(self, other)
        return self
        
    def __sub__(self, other):

        clone = copy(self)
        if other:
            return TLorentzVector.__sub__(clone, other)
        return clone

    def __isub__(self, other):
    
        TLorentzVector.__sub__(self, other)
        return self

    def __copy__(self):

        _copy = TLorentzVector(self)
        _copy.__class__ = self.__class__
        return _copy
