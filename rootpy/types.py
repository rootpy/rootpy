"""
Wrappers for basic types that are compatible with ROOT TTrees
"""
from array import array


class Variable(array):
    """This is the base class for all variables"""
    def __init__(self):

        array.__init__(self)
        self.default = 0

    def reset(self):
        """Reset the value to the default"""
        self[0] = self.default

    def clear(self):
        """Supplied to match the interface of ROOT.vector"""
        self.reset()

    @property
    def value(self):
        """The current value"""
        return self[0]

    def set(self, value):
        """Set the value"""
        if hasattr(value, '__getitem__'):
            self[0] = self.convert(value[0])
        else:
            self[0] = self.convert(value)

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        return "%s(%s) at %s" %  \
            (self.__class__.__name__, repr(self.value), id(self).__hex__())

    def __getitem__(self, i):

        return array.__getitem__(self, 0)

    def __setitem__(self, i, value):

        if hasattr(value, '__getitem__'):
            array.__setitem__(self, 0, value[0])
        else:
            array.__setitem__(self, 0, value)

    def __lt__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] < value[0]
        return self[0] < value

    def __le__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] <= value[0]
        return self[0] <= value

    def __eq__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] == value[0]
        return self[0] == value

    def __ne__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] != value[0]
        return self[0] != value

    def __gt__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] > value[0]
        return self[0] > value

    def __ge__(self, value):

        if hasattr(value, '__getitem__'):
            return self[0] >= value[0]
        return self[0] >= value

    def __nonzero__(self):

        return self[0] != 0

    def __add__(self, other):

        if hasattr(other, '__getitem__'):
            return self[0] + other[0]
        return self[0] + other

    def __radd__(self, other):

        return self + other

    def __sub__(self, other):

        if hasattr(other, '__getitem__'):
            return self[0] - other[0]
        return self[0] - other

    def __rsub__(self, other):

        return other - self[0]

    def __mul__(self, other):

        if hasattr(other, '__getitem__'):
            return self[0] * other[0]
        return self[0] * other

    def __rmul__(self, other):

        return self * other

    def __div__(self, other):

        if hasattr(other, '__getitem__'):
            return self[0] / other[0]
        return self[0] / other

    def __rdiv__(self, other):

        return other / self[0]

"""
ROOT:
- C : a character string terminated by the 0 character
- B : an 8 bit signed integer (Char_t)
- b : an 8 bit unsigned integer (UChar_t)
- S : a 16 bit signed integer (Short_t)
- s : a 16 bit unsigned integer (UShort_t)
- I : a 32 bit signed integer (Int_t)
- i : a 32 bit unsigned integer (UInt_t)
- F : a 32 bit floating point (Float_t)
- D : a 64 bit floating point (Double_t)
- L : a 64 bit signed integer (Long64_t)
- l : a 64 bit unsigned integer (ULong64_t)
- O : a boolean (Bool_t) (see note 1)

Python array:
Code    C Type          Python Type     Minimum Size in Bytes
'c'     char            character           1
'b'     signed char     int                 1
'B'     unsigned char   int                 1
'u'     Py_UNICODE      Unicode character   2 (see note 2)
'h'     signed short    int                 2
'H'     unsigned short  int                 2
'i'     signed int      int                 2
'I'     unsigned int    long                2
'l'     signed long     int                 4
'L'     unsigned long   long                4
'f'     float           float               4
'd'     double          float               8

Note 1: Use an unsigned type for array.array to represent a Bool_t

Note 2: The 'u' typecode corresponds to Python’s unicode character.
On narrow Unicode builds this is 2-bytes, on wide builds this is 4-bytes.
"""


class Char(Variable):
    """
    This is a variable containing a character type
    """

    # The ROOT character representation of the Boolean type
    type = 'C'

    def __new__(cls, default='\x00'):

        if type(default) is int:
            default = chr(default)
        return Variable.__new__(cls, 'c', [default])

    def __init__(self, default=False):

        Variable.__init__(self)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        if type(value) is int:
            return chr(value)
        return value


class Bool(Variable):
    """
    This is a variable containing a Boolean type
    """

    # The ROOT character representation of the Boolean type
    type = 'O'

    def __new__(cls, default=False):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'B', [int(bool(default))])

    def __init__(self, default=False):

        Variable.__init__(self)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        return int(bool(value))


class Int(Variable):
    """
    This is a variable containing an integer
    """

    # The ROOT character representation of the integer type
    type = 'I'

    def __new__(cls, default=0):

        return Variable.__new__(cls, 'i', [int(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.default = int(default)

    @staticmethod
    def convert(value):

        return int(value)


class UInt(Variable):
    """
    This is a variable containing an unsigned integer
    """

    # The ROOT character representation of the unsigned integer type
    type = 'i'

    def __new__(cls, default=0):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'I', [abs(long(default))])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        return abs(long(value))


class Float(Variable):
    """
    This is a variable containing a float
    """

    # The ROOT character representation of the float type
    type = 'F'

    def __new__(cls, default=0.):

        return Variable.__new__(cls, 'f', [float(default)])

    def __init__(self, default=0.):

        Variable.__init__(self)
        self.default = float(default)

    @staticmethod
    def convert(value):

        return float(value)


class Double(Variable):
    """
    This is a variable containing a float
    """

    # The ROOT character representation of the double type
    type = 'D'

    def __new__(cls, default=0.):

        return Variable.__new__(cls, 'd', [float(default)])

    def __init__(self, default=0.):

        Variable.__init__(self)
        self.default = float(default)

    @staticmethod
    def convert(value):

        return float(value)
