from .core import _repr_mixin, camelCaseMethods
from ROOT import TLorentzVector, TVector3, TVector2
from copy import copy

class _arithmetic_mixin(object):

    def __add__(self, other):
        
        clone = copy(self)
        if other:
            return self.__class__.__bases__[-1].__add__(clone, other)
        return clone
    
    def __radd__(self, other):

        clone = copy(self)
        if other:
            return self + other
        return clone
    
    def __iadd__(self, other):
        
        self.__class__.__bases__[-1].__add__(self, other)
        return self
        
    def __sub__(self, other):

        clone = copy(self)
        if other:
            return self.__class__.__bases__[-1].__sub__(clone, other)
        return clone
    
    def __rsub__(self, other):

        clone = copy(self)
        if other:
            raise AttributeError
        return clone
    
    def __isub__(self, other):
    
        self.__class__.__bases__[-1].__sub__(self, other)
        return self

    def __copy__(self):

        _copy = self.__class__.__bases__[-1](self)
        _copy.__class__ = self.__class__
        return _copy


@camelCaseMethods
class Vector2(_repr_mixin, _arithmetic_mixin, TVector2):

    def __repr__(self):

        return "%s(%f, %f)" % (self.__class__.__name__, self.X(), self.Y())


@camelCaseMethods
class Vector3(_repr_mixin, _arithmetic_mixin, TVector3):

    def __repr__(self):

        return "%s(%f, %f, %f)" % (self.__class__.__name__, self.X(), self.Y(), self.Z())
    
    def Angle(self, other):

        if isinstance(other, LorentzVector):
            return other.Angle(self)
        return TVector3.Angle(self, other)

@camelCaseMethods
class LorentzVector(_repr_mixin, _arithmetic_mixin, TLorentzVector):

    def __repr__(self):

        return "%s(%f, %f, %f, %f)" % (self.__class__.__name__, self.Px(), self.Py(), self.Pz(), self.E())
    
    def Angle(self, other):

        if isinstance(other, self.__class__):
            return TLorentzVector.Angle(self, other.Vect())
        return TLorentzVector.Angle(self, other)
