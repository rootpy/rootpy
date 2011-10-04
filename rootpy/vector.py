from .core import _repr_mixin, _copy_construct_mixin, camelCaseMethods, isbasictype
from .registry import register
from ROOT import TLorentzVector, TVector3, TVector2
from copy import copy


class _arithmetic_mixin(object):

    def __mul__(self, other):
      
        try: 
            prod = self.__class__.__bases__[-1].__mul__(self, other)
            if isinstance(prod, self.__class__.__bases__[-1]):
                prod.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return prod

    def __imul__(self, other):

        if isinstance(other, self.__class__):
            raise TypeError("Attemping to set vector to scalar quantity")
        try:
            prod = self * other
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        self = prod
        return self

    def __rmul__(self, other):
        
        try:
            return self * other
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (other.__class__.__name__, self.__class__.__name__))
    
    def __add__(self, other):
        
        try:
            clone = self.__class__.__bases__[-1].__add__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return clone

    def __radd__(self, other):
        
        if other == 0:
            return copy(self)
        raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % \
                (other.__class__.__name__, self.__class__.__name__))
    
    def __iadd__(self, other):
        
        try:
            _sum = self + other
        except TypeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        self = _sum
        return self

    def __sub__(self, other):
        
        try:
            clone = self.__class__.__bases__[-1].__sub__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return clone

    def __rsub__(self, other):
        
        if other == 0:
            return copy(self)
        raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" % \
                (other.__class__.__name__, self.__class__.__name__))
         
    def __isub__(self, other):
    
        try:
            diff = self - other
        except TypeError:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        self = diff
        return self

    def __copy__(self):

        _copy = self.__class__.__bases__[-1](self)
        _copy.__class__ = self.__class__
        return _copy


@camelCaseMethods
@register()
class Vector2(_arithmetic_mixin, _copy_construct_mixin, _repr_mixin, TVector2):

    def __repr__(self):

        return "%s(%f, %f)" % (self.__class__.__name__, self.X(), self.Y())
    
    def __mul__(self, other):
      
        if isinstance(other, self.__class__):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y()
        elif isbasictype(other):
            prod = Vector2(other * self.X(), other * self.Y())
        else:    
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return prod
    
    def __add__(self, other):
      
        if isinstance(other, TVector2):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y())
        else:    
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return _sum


@camelCaseMethods
@register()
class Vector3(_arithmetic_mixin, _copy_construct_mixin, _repr_mixin, TVector3):

    def __repr__(self):

        return "%s(%f, %f, %f)" % (self.__class__.__name__, self.X(), self.Y(), self.Z())
    
    def Angle(self, other):

        if isinstance(other, LorentzVector):
            return other.Angle(self)
        return TVector3.Angle(self, other)

    def __mul__(self, other):
      
        if isinstance(other, TVector3):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y() + \
                   self.Z() * other.Z()
        elif isbasictype(other):
            prod = Vector3(other * self.X(), other * self.Y(), other * self.Z())
        else:    
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return prod
    
    def __add__(self, other):
      
        if isinstance(other, TVector3):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y(),
                           self.Z() + other.Z())
        else:    
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % \
                (self.__class__.__name__, other.__class__.__name__))
        return _sum


@camelCaseMethods
@register()
class LorentzVector(_arithmetic_mixin, _copy_construct_mixin, _repr_mixin, TLorentzVector):

    def __repr__(self):

        return "%s(%f, %f, %f, %f)" % (self.__class__.__name__, self.Px(), self.Py(), self.Pz(), self.E())
    
    def Angle(self, other):

        if isinstance(other, TLorentzVector):
            return TLorentzVector.Angle(self, other.Vect())
        return TLorentzVector.Angle(self, other)

    def BoostVector(self):

        vector = TLorentzVector.BoostVector(self)
        vector.__class__ = Vector3
        return vector
