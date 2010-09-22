from array import array

class Variable(array):
    """
    This is the base class for all variables
    """        
    def reset(self):
        """
        Reset the value to the default 
        """
        self[0] = self.default

    def clear(self):
        """
        Supplied to match the interface of ROOT.vector
        """
        self.reset()
    
    def value(self):
        """
        The current value
        """
        return self[0]

    def set(self,value):
        """
        Set the value
        """
        self[0] = self.typename(value)
    
    def __str__(self):
        
        return self.__repr__()

    def __repr__(self):

        return "%s(%s) at %s"%(self.__class__.__name__,self[0],id(self).__hex__())

    def __getitem__(self,i):
        
        return array.__getitem__(self,0)

    def __setitem__(self,i,value):

        array.__setitem__(self,0,value)
    
    def __eq__(self,value):

        return self[0] == value

    def __ne__(self,value):

        return self[0] != value

#________________________________________________________________________

class Int(Variable):
    """
    This is a variable containing an integer
    """
    def __new__(cls, default=0):
        
        return Variable.__new__(cls,'i',[int(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.default = int(default)
        self.typename = int
    
    def type(self):
        """
        The character representation of the integer type
        """
        return 'I'

#________________________________________________________________________

class UInt(Variable):
    """
    This is a variable containing an unsigned integer
    """
    def __new__(cls, default=0):
        
        if default < 0:
            default = 0
        return Variable.__new__(cls,'I',[long(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        if default < 0:
            default = 0
        self.default = long(default)
        self.typename = long
    
    def type(self): return 'I'

#__________________________________________________________________________

class Float(Variable):
    """
    This is a variable containing a float
    """
    def __new__(cls, default=0.):
        
        return Variable.__new__(cls,'f',[float(default)])

    def __init__(self, default=0.):
        
        Variable.__init__(self)
        self.typename = float
        self.default = float(default)
    
    def type(self): return 'F'
