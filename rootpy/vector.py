from .core import _repr_mixin, camelCaseMethods, isbasictype
from ROOT import TLorentzVector, TVector3, TVector2
from copy import copy


class _arithmetic_mixin(object):

    def __mul__(self, other):
       
        prod = self.__class__.__bases__[-1].__mul__(self, other)
        if isinstance(prod, self.__class__.__bases__[-1]):
            prod.__class__ = self.__class__
        return prod

    def __imul__(self, other):

        if isinstance(other, self.__class__):
            raise TypeError("Attemping to set vector to scalar quantity")
        try:
            prod = self.__class__.__bases__[-1].__mul__(self, other)
        except TypeError:
            raise TypeError("Invalid operation")
        prod.__class__ = self.__class__
        self = prod
        return self

    def __rmul__(self, other):
        
        return self * other
    
    def __add__(self, other):
        
        try:
            clone = self.__class__.__bases__[-1].__add__(self, other)
        except TypeError:
            raise TypeError("Invalid operation")
        clone.__class__ = self.__class__
        return clone

    def __radd__(self, other):
        
        if other == 0:
            return copy(self)
        raise TypeError("Invalid operation")
    
    def __iadd__(self, other):
        
        try:
            _sum = self.__class__.__bases__[-1].__add__(self, other)
        except TypeError:
            raise TypeError("Invalid operation")
        _sum.__class__ = self.__class__
        self = _sum
        return self

    def __sub__(self, other):
        
        try:
            clone = self.__class__.__bases__[-1].__sub__(self, other)
        except TypeError:
            raise TypeError("Invalid operation")
        clone.__class__ = self.__class__
        return clone

    def __rsub__(self, other):
        
        if other == 0:
            return copy(self)
        raise TypeError("Invalid operation")
         
    def __isub__(self, other):
    
        try:
            diff = self.__class__.__bases__[-1].__sub__(self, other)
        except TypeError:
            raise TypeError("Invalid operation")
        diff.__class__ = self.__class__
        self = diff
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

    def BoostVector(self):

        vector = TLorentzVector.BoostVector(self)
        vector.__class__ = Vector3
        return vector
