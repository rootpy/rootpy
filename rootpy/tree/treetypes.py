# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Wrappers for basic types that are compatible with ROOT TTrees
"""
from __future__ import absolute_import

import itertools
from array import array
import sys
if sys.version_info[0] >= 3:
    long = int

from ..extern.six.moves import range
from .. import register

# only list Column subclasses here
__all__ = [
    'ObjectCol',
    'BoolCol',
    'BoolArrayCol',
    'CharCol',
    'CharArrayCol',
    'UCharCol',
    'UCharArrayCol',
    'ShortCol',
    'ShortArrayCol',
    'UShortCol',
    'UShortArrayCol',
    'IntCol',
    'IntArrayCol',
    'UIntCol',
    'UIntArrayCol',
    'LongCol',
    'LongArrayCol',
    'ULongCol',
    'ULongArrayCol',
    'FloatCol',
    'FloatArrayCol',
    'DoubleCol',
    'DoubleArrayCol',
]


class Column(object):
    _counter = itertools.count()

    def __init__(self, *args, **kwargs):
        self.idx = next(Column._counter)
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.type(*self.args, **self.kwargs)

    def __repr__(self):
        arg_params = ', '.join([str(a) for a in self.args])
        kwd_params = ', '.join(['{0}={1}'.format(name, value)
                                for name, value in self.kwargs.items()])
        params = []
        if arg_params:
            params.append(arg_params)
        if kwd_params:
            params.append(kwd_params)
        return "{0}({1})".format(
            self.__class__.__name__, ', '.join(params))

    def __str__(self):
        return repr(self)


class ObjectCol(Column):

    def __init__(self, cls, *args, **kwargs):
        self.type = cls
        Column.__init__(self, *args, **kwargs)


class Scalar(object):

    def clear(self):
        """Supplied to match the interface of ROOT.vector"""
        self.reset()


class BaseScalar(Scalar, array):
    """This is the base class for all variables"""

    def __init__(self, resetable=True):
        array.__init__(self)
        self.resetable = resetable

    def reset(self):
        """Reset the value to the default"""
        if self.resetable:
            self[0] = self.default

    @property
    def value(self):
        """The current value"""
        return self[0]

    def set(self, value):
        """Set the value"""
        if isinstance(value, BaseScalar):
            self[0] = self.convert(value.value)
        else:
            self[0] = self.convert(value)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{0}({1}) at {2}".format(
            self.__class__.__name__, repr(self.value), hex(id(self)))

    def __getitem__(self, i):
        return array.__getitem__(self, 0)

    def __setitem__(self, i, value):
        if isinstance(value, BaseScalar):
            array.__setitem__(self, 0, value.value)
        else:
            array.__setitem__(self, 0, value)

    def __lt__(self, value):
        if isinstance(value, BaseScalar):
            return self.value < value.value
        return self.value < value

    def __le__(self, value):
        if isinstance(value, BaseScalar):
            return self.value <= value.value
        return self.value <= value

    def __eq__(self, value):
        if isinstance(value, BaseScalar):
            return self.value == value.value
        return self.value == value

    def __ne__(self, value):
        if isinstance(value, BaseScalar):
            return self.value != value.value
        return self.value != value

    def __gt__(self, value):
        if isinstance(value, BaseScalar):
            return self.value > value.value
        return self.value > value

    def __ge__(self, value):
        if isinstance(value, BaseScalar):
            return self.value >= value.value
        return self.value >= value

    def __nonzero__(self):
        return self.value != 0

    __bool__ = __nonzero__

    def __add__(self, other):
        if isinstance(other, BaseScalar):
            return self.value + other.value
        return self.value + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, BaseScalar):
            return self.value - other.value
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        if isinstance(other, BaseScalar):
            return self.value * other.value
        return self.value * other

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, BaseScalar):
            return self.value / other.value
        return self.value / other

    def __rdiv__(self, other):
        return other / self.value


class Array(object):

    def __init__(self, resetable=True, length_name=None):
        self.resetable = resetable
        self.length_name = length_name

    def clear(self):
        """Supplied to match the interface of ROOT.vector"""
        self.reset()


class BaseArray(Array, array):
    """This is the base class for all array variables"""

    def __init__(self, **kwargs):
        array.__init__(self)
        Array.__init__(self, **kwargs)

    def reset(self):
        """Reset the value to the default"""
        if self.resetable:
            for i in range(len(self)):
                self[i] = self.default

    def set(self, other):
        for i, thing in enumerate(other):
            self[i] = self.convert(thing)
        for i in range(i + 1, len(self)):
            self[i] = self.default

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{0}[{1}] at {2}".format(
            self.__class__.__name__,
            ', '.join(map(str, self)),
            hex(id(self)))


class BaseChar(object):

    @property
    def value(self):
        return str(self.rstrip(b'\0').decode('ascii'))

    def __str__(self):
        return self.value

    def __repr__(self):
        return "{0}[{1}] at {2}".format(
            self.__class__.__name__,
            repr(str(self)),
            hex(id(self)))


class BaseCharScalar(BaseChar, Scalar, bytearray):
    """This is the base class for all char variables"""

    def __init__(self, resetable=True):
        bytearray.__init__(self, 2)
        self.resetable = resetable

    def reset(self):
        """Reset the value to the default"""
        if self.resetable:
            # reset to null bytes
            self[0] = 0

    def set(self, other):
        self[0] = other


class BaseCharArray(BaseChar, Array, bytearray):
    """This is the base class for all char array variables"""

    def __init__(self, length, **kwargs):
        if not isinstance(length, int):
            raise TypeError("char array length must be an int")
        if length < 2:
            raise ValueError(
                "char array length must be at least 2 "
                "to include null-termination")
        bytearray.__init__(self, length)
        Array.__init__(self, **kwargs)

    def reset(self):
        """Reset the value to the default"""
        if self.resetable:
            # reset to null bytes
            self[:] = bytearray(len(self))

    def set(self, other):
        # leave the null-termination untouched
        if len(other) >= len(self):
            raise ValueError(
                "string of length {0:d} is too long to "
                "fit in array of length {1:d} with null-termination".format(
                    len(other), len(self)))
        self[:len(other)] = other


@register(names=('B', 'Bool_t'), builtin=True)
class Bool(BaseScalar):
    """
    This is a variable containing a Boolean type
    """
    # The ROOT character representation of the Boolean type
    type = 'O'
    typename = 'Bool_t'

    def __new__(cls, default=False, **kwargs):
        return BaseScalar.__new__(cls, 'B', [Bool.convert(default)])

    def __init__(self, default=False, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Bool.convert(default)

    @classmethod
    def convert(cls, value):
        return int(bool(value))


class BoolCol(Column):
    type = Bool


@register(names=('B[]', 'Bool_t[]'), builtin=True)
class BoolArray(BaseArray):
    """
    This is an array of Booleans
    """
    # The ROOT character representation of the Boolean type
    type = 'O'
    typename = 'Bool_t'
    convert = Bool.convert

    def __new__(cls, length, default=False, **kwargs):
        return BaseArray.__new__(
            cls, 'B',
            [Bool.convert(default)] * length)

    def __init__(self, length, default=False, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Bool.convert(default)


class BoolArrayCol(Column):
    type = BoolArray


@register(names=('C', 'Char_t'), builtin=True)
class Char(BaseCharScalar):
    """
    This is a variable containing a character type
    """
    # The ROOT character representation of the char type
    type = 'C'
    typename = 'Char_t'


class CharCol(Column):
    type = Char


@register(names=('C[]', 'Char_t[]'), builtin=True)
class CharArray(BaseCharArray):
    """
    This is an array of characters
    """
    # The ROOT character representation of the char type
    type = 'C'
    typename = 'Char_t'
    scalar = Char


class CharArrayCol(Column):
    type = CharArray


@register(names=('UC', 'UChar_t'), builtin=True)
class UChar(BaseCharScalar):
    """
    This is a variable containing an unsigned character type
    """
    # The ROOT character representation of the unsigned char type
    type = 'c'
    typename = 'UChar_t'


class UCharCol(Column):
    type = UChar


@register(names=('UC[]', 'UChar_t[]'), builtin=True)
class UCharArray(BaseCharArray):
    """
    This is an array of unsigned characters
    """
    # The ROOT character representation of the unsigned char type
    type = 'c'
    typename = 'UChar_t'
    scalar = UChar


class UCharArrayCol(Column):
    type = UCharArray


@register(names=('S', 'Short_t'), builtin=True)
class Short(BaseScalar):
    """
    This is a variable containing an integer
    """
    # The ROOT character representation of the short type
    type = 'S'
    typename = 'Short_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'h', [Short.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Short.convert(default)

    @classmethod
    def convert(cls, value):
        return int(value)


class ShortCol(Column):
    type = Short


@register(names=('S[]', 'Short_t[]'), builtin=True)
class ShortArray(BaseArray):
    """
    This is an array of integers
    """
    # The ROOT character representation of the short type
    type = 'S'
    typename = 'Short_t'
    convert = Short.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'h',
            [Short.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Short.convert(default)


class ShortArrayCol(Column):
    type = ShortArray


@register(names=('US', 'UShort_t'), builtin=True)
class UShort(BaseScalar):
    """
    This is a variable containing a short
    """
    # The ROOT character representation of the unsigned short type
    type = 's'
    typename = 'UShort_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'H', [UShort.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = UShort.convert(default)

    @classmethod
    def convert(cls, value):
        if value < 0:
            raise ValueError(
                "Assigning negative value ({0:d}) "
                "to unsigned type".format(value))
        return int(value)


class UShortCol(Column):
    type = UShort


@register(names=('US[]', 'UShort_t[]'), builtin=True)
class UShortArray(BaseArray):
    """
    This is an array of unsigned shorts
    """
    # The ROOT character representation of the unsigned short type
    type = 's'
    typename = 'UShort_t'
    convert = UShort.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'H',
            [UShort.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = UShort.convert(default)


class UShortArrayCol(Column):
    type = UShortArray


@register(names=('I', 'Int_t'), builtin=True)
class Int(BaseScalar):
    """
    This is a variable containing an integer
    """
    # The ROOT character representation of the integer type
    type = 'I'
    typename = 'Int_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'i', [Int.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Int.convert(default)

    @classmethod
    def convert(cls, value):
        return int(value)


class IntCol(Column):
    type = Int


@register(names=('I[]', 'Int_t[]'), builtin=True)
class IntArray(BaseArray):
    """
    This is an array of integers
    """
    # The ROOT character representation of the integer type
    type = 'I'
    typename = 'Int_t'
    convert = Int.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'i',
            [Int.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Int.convert(default)


class IntArrayCol(Column):
    type = IntArray


@register(names=('UI', 'UInt_t'), builtin=True)
class UInt(BaseScalar):
    """
    This is a variable containing an unsigned integer
    """
    # The ROOT character representation of the unsigned integer type
    type = 'i'
    typename = 'UInt_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'I', [UInt.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = UInt.convert(default)

    @classmethod
    def convert(cls, value):
        if value < 0:
            raise ValueError(
                "Assigning negative value ({0:d}) "
                "to unsigned type".format(value))
        return long(value)


class UIntCol(Column):
    type = UInt


@register(names=('UI[]', 'UInt_t[]'), builtin=True)
class UIntArray(BaseArray):
    """
    This is an array of unsigned integers
    """
    # The ROOT character representation of the unsigned integer type
    type = 'i'
    typename = 'UInt_t'
    convert = UInt.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'I',
            [UInt.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = UInt.convert(default)


class UIntArrayCol(Column):
    type = UIntArray


@register(names=('L', 'Long64_t'), builtin=True)
class Long(BaseScalar):
    """
    This is a variable containing a long
    """
    # The ROOT character representation of the long type
    type = 'L'
    typename = 'Long64_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'l', [Long.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Long.convert(default)

    @classmethod
    def convert(cls, value):
        return long(value)


class LongCol(Column):
    type = Long


@register(names=('L[]', 'Long64_t[]'), builtin=True)
class LongArray(BaseArray):
    """
    This is an array of longs
    """
    # The ROOT character representation of the long type
    type = 'L'
    typename = 'Long64_t'
    convert = Long.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'l',
            [Long.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Long.convert(default)


class LongArrayCol(Column):
    type = LongArray


@register(names=('UL', 'ULong64_t'), builtin=True)
class ULong(BaseScalar):
    """
    This is a variable containing an unsigned long
    """
    # The ROOT character representation of the long type
    type = 'l'
    typename = 'ULong64_t'

    def __new__(cls, default=0, **kwargs):
        return BaseScalar.__new__(cls, 'L', [ULong.convert(default)])

    def __init__(self, default=0, **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = ULong.convert(default)

    @classmethod
    def convert(cls, value):
        if value < 0:
            raise ValueError(
                "Assigning negative value ({0:d}) "
                "to unsigned type".format(value))
        return long(value)


class ULongCol(Column):
    type = ULong


@register(names=('UL[]', 'ULong64_t[]'), builtin=True)
class ULongArray(BaseArray):
    """
    This is of unsigned longs
    """
    # The ROOT character representation of the long type
    type = 'l'
    typename = 'ULong64_t'
    convert = ULong.convert

    def __new__(cls, length, default=0, **kwargs):
        return BaseArray.__new__(
            cls, 'L',
            [ULong.convert(default)] * length)

    def __init__(self, length, default=0, **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = ULong.convert(default)


class ULongArrayCol(Column):
    type = ULongArray


@register(names=('F', 'Float_t'), builtin=True)
class Float(BaseScalar):
    """
    This is a variable containing a float
    """
    # The ROOT character representation of the float type
    type = 'F'
    typename = 'Float_t'

    def __new__(cls, default=0., **kwargs):
        return BaseScalar.__new__(cls, 'f', [Float.convert(default)])

    def __init__(self, default=0., **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Float.convert(default)

    @classmethod
    def convert(cls, value):
        return float(value)


class FloatCol(Column):
    type = Float


@register(names=('F[]', 'Float_t[]'), builtin=True)
class FloatArray(BaseArray):
    """
    This is an array of floats
    """
    # The ROOT character representation of the float type
    type = 'F'
    typename = 'Float_t'
    convert = Float.convert

    def __new__(cls, length, default=0., **kwargs):
        return BaseArray.__new__(
            cls, 'f',
            [Float.convert(default)] * length)

    def __init__(self, length, default=0., **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Float.convert(default)


class FloatArrayCol(Column):
    type = FloatArray


@register(names=('D', 'Double_t'), builtin=True)
class Double(BaseScalar):
    """
    This is a variable containing a double
    """
    # The ROOT character representation of the double type
    type = 'D'
    typename = 'Double_t'

    def __new__(cls, default=0., **kwargs):
        return BaseScalar.__new__(cls, 'd', [Double.convert(default)])

    def __init__(self, default=0., **kwargs):
        BaseScalar.__init__(self, **kwargs)
        self.default = Double.convert(default)

    @classmethod
    def convert(cls, value):
        return float(value)


class DoubleCol(Column):
    type = Double


@register(names=('D[]', 'Double_t[]'), builtin=True)
class DoubleArray(BaseArray):
    """
    This is an array of doubles
    """
    # The ROOT character representation of the double type
    type = 'D'
    typename = 'Double_t'
    convert = Double.convert

    def __new__(cls, length, default=0., **kwargs):
        return BaseArray.__new__(
            cls, 'd',
            [Double.convert(default)] * length)

    def __init__(self, length, default=0., **kwargs):
        BaseArray.__init__(self, **kwargs)
        self.default = Double.convert(default)


class DoubleArrayCol(Column):
    type = DoubleArray


# ROOT type codes:
root_type_codes = '''\
O       a boolean (Bool_t) (see note 1)
B       an 8 bit signed integer (Char_t)
b       an 8 bit unsigned integer (UChar_t)
S       a 16 bit signed integer (Short_t)
s       a 16 bit unsigned integer (UShort_t)
I       a 32 bit signed integer (Int_t)
i       a 32 bit unsigned integer (UInt_t)
L       a 64 bit signed integer (Long64_t)
l       a 64 bit unsigned integer (ULong64_t)
F       a 32 bit floating point (Float_t)
D       a 64 bit floating point (Double_t)\
'''

root_type_codes = [line.split()[0] for line in root_type_codes.split('\n')]

# ROOT type names:
root_type_names = '''\
Bool_t
Char_t
UChar_t
Short_t
UShort_t
Int_t
UInt_t
Long64_t
ULong64_t
Float_t
Double_t\
'''

root_type_names = [line.split()[0] for line in root_type_names.split('\n')]

# Python array:
python_codes = '''\
B       unsigned char   int                 1 (used as boolean)
b       signed char     int                 1
B       unsigned char   int                 1
h       signed short    int                 2
H       unsigned short  int                 2
i       signed int      int                 2
I       unsigned int    long                2
l       signed long     int                 4
L       unsigned long   long                4
f       float           float               4
d       double          float               8\
'''

python_codes = [line.split()[0] for line in python_codes.split('\n')]

# Python NumPy array:
numpy_codes = '''\
b       Boolean
i1      Char
u1      Unsigned Char
i2      Short Integer
u2      Unsigned Short integer
i4      Integer
u4      Unsigned integer
i8      Long Integer
u8      Unsigned Long integer
f4      Floating point
f8      Double Floating point\
'''

numpy_codes = [line.split()[0] for line in numpy_codes.split('\n')]


def convert(origin, target, type):
    """
    convert type from origin to target
    origin/target must be ROOTCODE, ROOTNAME, ARRAY, or NUMPY
    """
    _origin = origin.upper()
    if _origin == 'ROOTCODE':
        _origin = root_type_codes
    elif _origin == 'ROOTNAME':
        _origin = root_type_names
    elif _origin == 'ARRAY':
        _origin = python_codes
    elif _origin == 'NUMPY':
        _origin = numpy_codes
    else:
        raise ValueError("{0} is not a valid type".format(origin))

    _target = target.upper()
    if _target == 'ROOTCODE':
        _target = root_type_codes
    elif _target == 'ROOTNAME':
        _target = root_type_names
    elif _target == 'ARRAY':
        _target = python_codes
    elif _target == 'NUMPY':
        _target = numpy_codes
    else:
        raise ValueError("{0} is not a valid type".format(target))

    if type not in _origin:
        raise ValueError("{0} is not a valid {1} type".format(type, origin))

    return _target[_origin.index(type)]
