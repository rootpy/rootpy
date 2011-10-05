# ROOT type codes:
root_type_codes = \
'''\
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
root_type_names = \
'''\
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
python_codes = \
'''\
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
numpy_codes = \
'''\
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
        raise ValueError("%s is not a valid type" % origin)
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
        raise ValueError("%s is not a valid type" % target)
    try:
        index = _origin.index(type)
    except:
        raise ValueError("%s is not a valid %s type" % (type, origin))
    return _target[index]
