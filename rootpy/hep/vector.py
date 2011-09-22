from ROOT import TLorentzVector
from ..core import NamelessConstructorObject

class FourVector(NamelessConstructorObject, TLorentzVector):

    def __init__(self, *args, **kwargs):

        NamelessConstructorObject.__init__(self, *args, **kwargs)

    def __add__(self, other):
        
        clone = self.Clone() 
        if other:
            return TLorentzVector.__add__(clone, other)
        return clone
    
    def __iadd__(self, other):
        
        TLorentzVector.__add__(self, other)
        return self
        
    def __sub__(self, other):

        clone = self.Clone()
        if other:
            return TLorentzVector.__sub__(clone, other)
        return clone

    def __isub__(self, other):
    
        TLorentzVector.__sub__(self, other)
        return self
