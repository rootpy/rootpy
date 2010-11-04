from array import array

class Variable(array):
    """This is the base class for all variables"""        
    def __init__(self):

        self.typename = float
        self.default = 0
        
    def reset(self):
        """Reset the value to the default"""
        self[0] = self.default

    def clear(self):
        """Supplied to match the interface of ROOT.vector"""
        self.reset()
    
    def value(self):
        """The current value"""
        return self[0]

    def set(self,value):
        """Set the value"""
        self[0] = self.typename(value)
    
    def __str__(self):
        
        return self.__repr__()

    def __repr__(self):

        return "%s(%s) at %s"%(self.__class__.__name__,self[0],id(self).__hex__())

    def __getitem__(self,i):
        
        return array.__getitem__(self,0)

    def __setitem__(self,i,value):

        array.__setitem__(self,0,value)

    def __lt__(self,value):

        return self[0] < value

    def __le__(self,value):

        return self[0] <= value
    
    def __eq__(self,value):

        return self[0] == value

    def __ne__(self,value):

        return self[0] != value

    def __gt__(self,value):

        return self[0] > value
    
    def __ge__(self,value):

        return self[0] >= value

"""
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
- O : a boolean (Bool_t)
"""

#________________________________________________________________________

class Int(Variable):
    """
    This is a variable containing an integer
    """
    def __new__(cls, default=0):
        
        return Variable.__new__(cls,'i',[int(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.typename = int
        self.default = int(default)
    
    def type(self):
        """The character representation of the integer type"""
        return 'I'

#________________________________________________________________________

class UInt(Variable):
    """This is a variable containing an unsigned integer"""
    def __new__(cls, default=0):
        
        if default < 0:
            default = 0
        return Variable.__new__(cls,'I',[long(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        if default < 0:
            default = 0
        self.typename = long
        self.default = long(default)
    
    def type(self):
        """The character representation of the unsigned integer type"""
        return 'i'

#__________________________________________________________________________

class Float(Variable):
    """This is a variable containing a float"""
    def __new__(cls, default=0.):
        
        return Variable.__new__(cls,'f',[float(default)])

    def __init__(self, default=0.):
        
        Variable.__init__(self)
        self.typename = float
        self.default = float(default)
    
    def type(self):
        """The character representation of the float type"""
        return 'F'
