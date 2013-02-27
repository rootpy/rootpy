# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from ... import QROOT
from ...core import (_repr_mixin, _copy_construct_mixin,
                     _resetable_mixin, isbasictype)
from ...decorators import snake_case_methods
import ROOT
from copy import copy


class _arithmetic_mixin:

    def __mul__(self, other):

        try:
            prod = self.__class__.__bases__[-1].__mul__(self, other)
            if isinstance(prod, self.__class__.__bases__[-1]):
                prod.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return prod

    def __imul__(self, other):

        if isinstance(other, self.__class__):
            raise TypeError("Attemping to set vector to scalar quantity")
        try:
            prod = self * other
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        self = prod
        return self

    def __rmul__(self, other):

        try:
            return self * other
        except TypeError:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (other.__class__.__name__, self.__class__.__name__))

    def __add__(self, other):

        if other == 0:
            return copy(self)
        try:
            clone = self.__class__.__bases__[-1].__add__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return clone

    def __radd__(self, other):

        if other == 0:
            return copy(self)
        raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                (other.__class__.__name__, self.__class__.__name__))

    def __iadd__(self, other):

        try:
            _sum = self + other
        except TypeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        self = _sum
        return self

    def __sub__(self, other):

        if other == 0:
            return copy(self)
        try:
            clone = self.__class__.__bases__[-1].__sub__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return clone

    def __rsub__(self, other):

        if other == 0:
            return copy(self)
        raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                (other.__class__.__name__, self.__class__.__name__))

    def __isub__(self, other):

        try:
            diff = self - other
        except TypeError:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        self = diff
        return self

    def __copy__(self):

        _copy = self.__class__.__bases__[-1](self)
        _copy.__class__ = self.__class__
        return _copy


@snake_case_methods
class Vector2(_arithmetic_mixin, _copy_construct_mixin,
              _resetable_mixin, _repr_mixin, QROOT.TVector2):

    def __repr__(self):

        return "%s(%f, %f)" % (self.__class__.__name__, self.X(), self.Y())

    def __mul__(self, other):

        if isinstance(other, self.__class__):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y()
        elif isbasictype(other):
            prod = Vector2(other * self.X(), other * self.Y())
        else:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return prod

    def __add__(self, other):

        if isinstance(other, ROOT.TVector2):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y())
        else:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return _sum


@snake_case_methods
class Vector3(_arithmetic_mixin, _copy_construct_mixin,
              _repr_mixin, _resetable_mixin, QROOT.TVector3):

    def __repr__(self):

        return("%s(%f, %f, %f)" %
               (self.__class__.__name__, self.X(), self.Y(), self.Z()))

    def Angle(self, other):

        if isinstance(other, LorentzVector):
            return other.Angle(self)
        return ROOT.TVector3.Angle(self, other)

    def __mul__(self, other):

        if isinstance(other, ROOT.TVector3):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y() + \
                   self.Z() * other.Z()
        elif isbasictype(other):
            prod = Vector3(other * self.X(), other * self.Y(), other * self.Z())
        else:
            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return prod

    def __add__(self, other):

        if isinstance(other, ROOT.TVector3):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y(),
                           self.Z() + other.Z())
        else:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return _sum

    def __sub__(self, other):

        if isinstance(other, ROOT.TVector3):
            _dif = Vector3(self.X() - other.X(),
                           self.Y() - other.Y(),
                           self.Z() - other.Z())
        else:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                (self.__class__.__name__, other.__class__.__name__))
        return _dif


@snake_case_methods
class LorentzVector(_arithmetic_mixin, _copy_construct_mixin,
                    _repr_mixin, _resetable_mixin, QROOT.TLorentzVector):

    def __repr__(self):

        return("%s(%f, %f, %f, %f)" %
               (self.__class__.__name__, self.Px(), self.Py(), self.Pz(), self.E()))

    def Angle(self, other):

        if isinstance(other, ROOT.TLorentzVector):
            return ROOT.TLorentzVector.Angle(self, other.Vect())
        return ROOT.TLorentzVector.Angle(self, other)

    def BoostVector(self):

        vector = ROOT.TLorentzVector.BoostVector(self)
        vector.__class__ = Vector3
        return vector


@snake_case_methods
class Rotation(_arithmetic_mixin, _copy_construct_mixin,
               _repr_mixin, _resetable_mixin, QROOT.TRotation):

    def __repr__(self):

        return "[[%f, %f, %f],\n" \
               " [%f, %f, %f],\n" \
               " [%f, %f, %f]]" % (self.XX(), self.XY(), self.XZ(),
                                   self.YX(), self.YY(), self.YZ(),
                                   self.ZX(), self.ZY(), self.ZZ())


@snake_case_methods
class LorentzRotation(_arithmetic_mixin, _copy_construct_mixin,
                      _repr_mixin, _resetable_mixin, QROOT.TLorentzRotation):

    def __repr__(self):

        return "[[%f, %f, %f, %f],\n" \
               " [%f, %f, %f, %f],\n" \
               " [%f, %f, %f, %f],\n" \
               " [%f, %f, %f, %f]]" % (self.XX(), self.XY(), self.XZ(), self.XT(),
                                       self.YX(), self.YY(), self.YZ(), self.YT(),
                                       self.ZX(), self.ZY(), self.ZZ(), self.ZT(),
                                       self.TX(), self.TY(), self.TZ(), self.TT())
