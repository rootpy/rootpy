# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

import numbers
from copy import copy

from . import QROOT
from .base import Object
from .decorators import snake_case_methods

__all__ = [
    'Vector2',
    'Vector3',
    'LorentzVector',
    'Rotation',
    'LorentzRotation',
]


class _arithmetic_mixin:

    def __mul__(self, other):
        try:
            prod = self.__class__.__bases__[-1].__mul__(self, other)
            if isinstance(prod, self.__class__.__bases__[-1]):
                prod.__class__ = self.__class__
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return prod

    def __imul__(self, other):
        if isinstance(other, self.__class__):
            raise TypeError("Attemping to set vector to scalar quantity")
        try:
            prod = self * other
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        self = prod
        return self

    def __rmul__(self, other):
        try:
            return self * other
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    other.__class__.__name__, self.__class__.__name__))

    def __add__(self, other):
        if other == 0:
            return copy(self)
        try:
            clone = self.__class__.__bases__[-1].__add__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return clone

    def __radd__(self, other):
        if other == 0:
            return copy(self)
        raise TypeError(
            "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                other.__class__.__name__, self.__class__.__name__))

    def __iadd__(self, other):
        try:
            _sum = self + other
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        self = _sum
        return self

    def __sub__(self, other):
        if other == 0:
            return copy(self)
        try:
            clone = self.__class__.__bases__[-1].__sub__(self, other)
            clone.__class__ = self.__class__
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for -: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return clone

    def __rsub__(self, other):
        if other == 0:
            return copy(self)
        raise TypeError(
                "unsupported operand type(s) for -: '{0}' and '{1}'".format(
                    other.__class__.__name__, self.__class__.__name__))

    def __isub__(self, other):
        try:
            diff = self - other
        except TypeError:
            raise TypeError(
                "unsupported operand type(s) for -: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        self = diff
        return self

    def __copy__(self):
        _copy = self.__class__.__bases__[-1](self)
        _copy.__class__ = self.__class__
        return _copy


@snake_case_methods
class Vector2(_arithmetic_mixin, Object, QROOT.TVector2):
    """
    A subclass of `ROOT.TVector2 <http://root.cern.ch/root/html/TVector2.html>`_.

    Examples
    --------

    >>> from rootpy.vector import Vector2
    >>> vect = Vector2(2, 4)
    >>> vect
    Vector2(x=2.000000, y=4.000000)

    """
    _ROOT = QROOT.TVector2

    @property
    def x(self):
        return self.X()

    @property
    def y(self):
        return self.Y()

    def __getitem__(self, i):
        if i == 0:
            return self.X()
        elif i == 1:
            return self.Y()
        raise IndexError("index {0:d} out of bounds".format(i))

    def __repr__(self):
        return '{0}(x={1:f}, y={2:f})'.format(
            self.__class__.__name__, self.X(), self.Y())

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y()
        elif isinstance(other, numbers.Real):
            prod = Vector2(other * self.X(), other * self.Y())
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return prod

    def __add__(self, other):
        if isinstance(other, ROOT.TVector2):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y())
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return _sum


@snake_case_methods
class Vector3(_arithmetic_mixin, Object, QROOT.TVector3):
    """
    A subclass of `ROOT.TVector3 <http://root.cern.ch/root/html/TVector3.html>`_.

    Examples
    --------

    >>> from rootpy.vector import Vector3
    >>> vect = Vector3(1, 2, 3)
    >>> vect
    Vector3(x=1.000000, y=2.000000, z=3.000000)

    """
    _ROOT = QROOT.TVector3

    @property
    def x(self):
        return self.X()

    @property
    def y(self):
        return self.Y()

    @property
    def z(self):
        return self.Z()

    def __getitem__(self, i):
        if i == 0:
            return self.X()
        elif i == 1:
            return self.Y()
        elif i == 2:
            return self.Z()
        raise IndexError("index {0:d} out of bounds".format(i))

    def __repr__(self):
        return '{0}(x={1:f}, y={2:f}, z={3:f})'.format(
            self.__class__.__name__, self.X(), self.Y(), self.Z())

    def Angle(self, other):
        if isinstance(other, LorentzVector):
            return other.Angle(self)
        return ROOT.TVector3.Angle(self, other)

    def __mul__(self, other):
        if isinstance(other, ROOT.TVector3):
            prod = self.X() * other.X() + \
                   self.Y() * other.Y() + \
                   self.Z() * other.Z()
        elif isinstance(other, numbers.Real):
            prod = Vector3(other * self.X(), other * self.Y(), other * self.Z())
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return prod

    def __add__(self, other):
        if isinstance(other, ROOT.TVector3):
            _sum = Vector3(self.X() + other.X(),
                           self.Y() + other.Y(),
                           self.Z() + other.Z())
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return _sum

    def __sub__(self, other):
        if isinstance(other, ROOT.TVector3):
            _dif = Vector3(self.X() - other.X(),
                           self.Y() - other.Y(),
                           self.Z() - other.Z())
        else:
            raise TypeError(
                "unsupported operand type(s) for -: '{0}' and '{1}'".format(
                    self.__class__.__name__, other.__class__.__name__))
        return _dif


@snake_case_methods
class LorentzVector(_arithmetic_mixin, Object, QROOT.TLorentzVector):
    """
    A subclass of `ROOT.TLorentzVector <http://root.cern.ch/root/html/TLorentzVector.html>`_.

    Examples
    --------

    >>> from rootpy.vector import LorentzVector
    >>> vect = LorentzVector(1, 2, 3, 4)
    >>> vect
    LorentzVector(px=1.000000, py=2.000000, pz=3.000000, E=4.000000)

    """
    _ROOT = QROOT.TLorentzVector

    @property
    def px(self):
        return self.Px()

    @property
    def py(self):
        return self.Py()

    @property
    def pz(self):
        return self.Pz()

    @property
    def e(self):
        return self.E()

    def __getitem__(self, i):
        if i == 0:
            return self.Px()
        elif i == 1:
            return self.Py()
        elif i == 2:
            return self.Pz()
        elif i == 3:
            return self.E()
        raise IndexError("index {0:d} out of bounds".format(i))

    def __repr__(self):
        return "{0}(px={1:f}, py={2:f}, pz={3:f}, E={4:f})".format(
            self.__class__.__name__,
            self.Px(), self.Py(), self.Pz(), self.E())

    def Angle(self, other):
        if isinstance(other, ROOT.TLorentzVector):
            return ROOT.TLorentzVector.Angle(self, other.Vect())
        return ROOT.TLorentzVector.Angle(self, other)

    def BoostVector(self):
        vector = ROOT.TLorentzVector.BoostVector(self)
        vector.__class__ = Vector3
        return vector


@snake_case_methods
class Rotation(_arithmetic_mixin, Object, QROOT.TRotation):
    """
    A subclass of `ROOT.TRotation <http://root.cern.ch/root/html/TRotation.html>`_.
    """
    _ROOT = QROOT.TRotation

    def __repr__(self):
        return ("[[{0:f}, {1:f}, {2:f}],\n"
                " [{3:f}, {4:f}, {5:f}],\n"
                " [{6:f}, {7:f}, {8:f}]]").format(
                    self.XX(), self.XY(), self.XZ(),
                    self.YX(), self.YY(), self.YZ(),
                    self.ZX(), self.ZY(), self.ZZ())


@snake_case_methods
class LorentzRotation(_arithmetic_mixin, Object, QROOT.TLorentzRotation):
    """
    A subclass of `ROOT.TLorentzRotation <http://root.cern.ch/root/html/TLorentzRotation.html>`_.
    """
    _ROOT = QROOT.TLorentzRotation

    def __repr__(self):
        return ("[[{0:f},  {1:f},  {2:f},  {3:f}],\n"
                " [{4:f},  {5:f},  {6:f},  {7:f}],\n"
                " [{8:f},  {9:f},  {10:f},  {11:f}],\n"
                " [{12:f},  {13:f},  {14:f},  {15:f}]]").format(
                    self.XX(), self.XY(), self.XZ(), self.XT(),
                    self.YX(), self.YY(), self.YZ(), self.YT(),
                    self.ZX(), self.ZY(), self.ZZ(), self.ZT(),
                    self.TX(), self.TY(), self.TZ(), self.TT())
