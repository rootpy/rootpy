# -*- coding: iso-8859-1 -*-

"""
Wrappers for basic types that are compatible with ROOT TTrees
"""
from array import array
from ..registry import register
from .convert import convert 
import ROOT


class Column(object):

    def __init__(self, *args, **kwargs):
        
        self.args = args
        self.kwargs = kwargs

    def __call__(self):

        return self.type(*self.args, **self.kwargs)
    
    def __repr__(self):

        arg_params = ', '.join([str(a) for a in self.args])
        kwd_params = ', '.join(['%s=%s' % (name, value)
                                for name, value in self.kwargs.items()])
        params = []
        if arg_params:
            params.append(arg_params)
        if kwd_params:
            params.append(kwd_params)
        return "%s(%s)" % (self.__class__.__name__, ', '.join(params))

    def __str__(self):

        return repr(self)


class ObjectCol(Column):

    def __init__(self, cls, *args, **kwargs):

        self.type = cls
        Column.__init__(self, *args, **kwargs)


class Variable(array):
    """This is the base class for all variables"""
    
    def __init__(self, resetable=True):

        array.__init__(self)
        self.resetable = resetable

    def reset(self):
        """Reset the value to the default"""
        if self.resetable:
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
        if isinstance(value, Variable):
            self[0] = self.convert(value.value)
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

        if isinstance(value, Variable):
            array.__setitem__(self, 0, value.value)
        else:
            array.__setitem__(self, 0, value)

    def __lt__(self, value):

        if isinstance(value, Variable):
            return self.value < value.value
        return self.value < value

    def __le__(self, value):

        if isinstance(value, Variable):
            return self.value <= value.value
        return self.value <= value

    def __eq__(self, value):

        if isinstance(value, Variable):
            return self.value == value.value
        return self.value == value

    def __ne__(self, value):

        if isinstance(value, Variable):
            return self.value != value.value
        return self.value != value

    def __gt__(self, value):

        if isinstance(value, Variable):
            return self.value > value.value
        return self.value > value

    def __ge__(self, value):

        if isinstance(value, Variable):
            return self.value >= value.value
        return self.value >= value

    def __nonzero__(self):

        return self.value != 0

    def __add__(self, other):

        if isinstance(other, Variable):
            return self.value + other.value
        return self.value + other

    def __radd__(self, other):

        return self + other

    def __sub__(self, other):

        if isinstance(other, Variable):
            return self.value - other.value
        return self.value - other

    def __rsub__(self, other):

        return other - self.value

    def __mul__(self, other):

        if isinstance(other, Variable):
            return self.value * other.value
        return self.value * other

    def __rmul__(self, other):

        return self * other

    def __div__(self, other):

        if isinstance(other, Variable):
            return self.value / other.value
        return self.value / other

    def __rdiv__(self, other):

        return other / self.value


@register(names=('C', 'CHAR_T'), builtin=True)
class Char(Variable):
    """
    This is a variable containing a character type
    """

    # The ROOT character representation of the Boolean type
    type = 'C'
    typename = 'Char_t'

    def __new__(cls, default='\x00', **kwargs):

        if type(default) is int:
            default = chr(default)
        return Variable.__new__(cls, 'c', [default])

    def __init__(self, default=False, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        if type(value) is int:
            return chr(value)
        return value


class CharCol(Column):
    type = Char


@register(names=('B', 'BOOL_T'), builtin=True)
class Bool(Variable):
    """
    This is a variable containing a Boolean type
    """

    # The ROOT character representation of the Boolean type
    type = 'O'
    typename = 'Bool_t'

    def __new__(cls, default=False, **kwargs):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'B', [int(bool(default))])

    def __init__(self, default=False, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        return int(bool(value))


class BoolCol(Column):
    type = Bool


@register(names=('I', 'INT_T'), builtin=True)
class Int(Variable):
    """
    This is a variable containing an integer
    """

    # The ROOT character representation of the integer type
    type = 'I'
    typename = 'Int_t'

    def __new__(cls, default=0, **kwargs):

        return Variable.__new__(cls, 'i', [int(default)])

    def __init__(self, default=0, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = int(default)

    @staticmethod
    def convert(value):

        return int(value)


class IntCol(Column):
    type = Int


@register(names=('UI', 'UINT_T'), builtin=True)
class UInt(Variable):
    """
    This is a variable containing an unsigned integer
    """

    # The ROOT character representation of the unsigned integer type
    type = 'i'
    typename = 'UInt_t'

    def __new__(cls, default=0, **kwargs):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'I', [long(default)])

    def __init__(self, default=0, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):
        
        if value < 0:
            raise ValueError("Assigning negative value (%i) to unsigned type" % value)
        return long(value)


class UIntCol(Column):
    type = UInt


@register(names=('L', 'LONG64_T'), builtin=True)
class Long(Variable):
    """
    This is a variable containing a long
    """

    # The ROOT character representation of the long type
    type = 'L'
    typename = 'Long64_t'

    def __new__(cls, default=0, **kwargs):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'l', [long(default)])

    def __init__(self, default=0, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        return long(value)


class LongCol(Column):
    type = Long


@register(names=('UL', 'ULONG64_T'), builtin=True)
class ULong(Variable):
    """
    This is a variable containing a long
    """

    # The ROOT character representation of the long type
    type = 'l'
    typename = 'ULong64_t'

    def __new__(cls, default=0, **kwargs):

        if default < 0:
            default = 0
        return Variable.__new__(cls, 'L', [long(default)])

    def __init__(self, default=0, **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = self.convert(default)

    @staticmethod
    def convert(value):

        if value < 0:
            raise ValueError("Assigning negative value (%i) to unsigned type" % value)
        return long(value)


class ULongCol(Column):
    type = ULong


@register(names=('F', 'FLOAT_T'), builtin=True)
class Float(Variable):
    """
    This is a variable containing a float
    """

    # The ROOT character representation of the float type
    type = 'F'
    typename = 'Float_t'

    def __new__(cls, default=0., **kwargs):

        return Variable.__new__(cls, 'f', [float(default)])

    def __init__(self, default=0., **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = float(default)

    @staticmethod
    def convert(value):

        return float(value)


class FloatCol(Column):
    type = Float


@register(names=('D', 'DOUBLE_T'), builtin=True)
class Double(Variable):
    """
    This is a variable containing a double
    """

    # The ROOT character representation of the double type
    type = 'D'
    typename = 'Double_t'

    def __new__(cls, default=0., **kwargs):

        return Variable.__new__(cls, 'd', [float(default)])

    def __init__(self, default=0., **kwargs):

        Variable.__init__(self, **kwargs)
        self.default = float(default)

    @staticmethod
    def convert(value):

        return float(value)


class DoubleCol(Column):
    type = Double


"""
generate dictionaries for commonly used types not included in ROOT
"""
from ..classfactory import generate

generate('vector<vector<float> >', '<vector>')
generate('vector<vector<int> >', '<vector>')
generate('vector<vector<unsigned int> >', '<vector>')
generate('vector<vector<long> >', '<vector>')
generate('vector<vector<unsigned long> >', '<vector>')
generate('vector<vector<double> >', '<vector>')
generate('vector<vector<string> >')
generate('map<string,string>', '<map>;<string>')
generate('vector<TLorentzVector>', 'TLorentzVector.h')
generate('vector<vector<TLorentzVector> >', 'TLorentzVector.h')


"""
Register builtin types:
"""
register(builtin=True, names=('VS',    'VECTOR<SHORT>'))                                (ROOT.vector('short'))
register(builtin=True, names=('VUS',   'VECTOR<UNSIGNED SHORT>'))                       (ROOT.vector('unsigned short'))
register(builtin=True, names=('VI',    'VECTOR<INT>'), demote='I')                      (ROOT.vector('int'))
register(builtin=True, names=('VUI',   'VECTOR<UNSIGNED INT>'), demote='UI')            (ROOT.vector('unsigned int'))
register(builtin=True, names=('VL',    'VECTOR<LONG>'), demote='L')                     (ROOT.vector('long'))
register(builtin=True, names=('VUL',   'VECTOR<UNSIGNED LONG>'), demote='UL')           (ROOT.vector('unsigned long'))
register(builtin=True, names=('VF',    'VECTOR<FLOAT>'), demote='F')                    (ROOT.vector('float'))
register(builtin=True, names=('VD',    'VECTOR<DOUBLE>'), demote='D')                   (ROOT.vector('double'))
register(builtin=True, names=('VVI',   'VECTOR<VECTOR<INT> >'), demote='VI')            (ROOT.vector('vector<int>'))
register(builtin=True, names=('VVUI',  'VECTOR<VECTOR<UNSIGNED INT> >'), demote='VUI')  (ROOT.vector('vector<unsigned int>'))
register(builtin=True, names=('VVL',   'VECTOR<VECTOR<LONG> >'), demote='VL')           (ROOT.vector('vector<long>'))
register(builtin=True, names=('VVUL',  'VECTOR<VECTOR<UNSIGNED LONG> >'), demote='VUL') (ROOT.vector('vector<unsigned long>'))
register(builtin=True, names=('VVF',   'VECTOR<VECTOR<FLOAT> >'), demote='VF')          (ROOT.vector('vector<float>'))
register(builtin=True, names=('VVD',   'VECTOR<VECTOR<DOUBLE> >'), demote='VD')         (ROOT.vector('vector<double>'))
register(builtin=True, names=('VVSTR', 'VECTOR<VECTOR<STRING> >'), demote='VSTR')       (ROOT.vector('vector<string>'))
register(builtin=True, names=('VSTR',  'VECTOR<STRING>'))                               (ROOT.vector('string'))
register(builtin=True, names=('MSI',   'MAP<STRING,INT>'))                              (ROOT.map('string,int'))
register(builtin=True, names=('MSF',   'MAP<STRING,FLOAT>'))                            (ROOT.map('string,float'))
register(builtin=True, names=('MSS',   'MAP<STRING,STRING>'))                           (ROOT.map('string,string'))

register(builtin=True, names=('VECTOR<TLORENTZVECTOR>',))(ROOT.vector('TLorentzVector'))

