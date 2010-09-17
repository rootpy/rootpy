from array import array

class Variable(array):
        
    def reset(self):
        
        self[0] = self.default

    def clear(self):

        self.reset()
    
    def value(self):
        
        return self[0]

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
    
    def __new__(cls, default=0):
        
        return Variable.__new__(cls,'i',[int(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.default = int(default)
    
    def set(self, value):
    
        self[0] = int(value)
    
    def type(self): return 'I'

#________________________________________________________________________

class UInt(Variable):
    
    def __new__(cls, default=0):
        
        if default < 0:
            default = 0
        return Variable.__new__(cls,'I',[long(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        if default < 0:
            default = 0
        self.default = long(default)
    
    def set(self, value):
    
        self[0] = long(value)
    
    def type(self): return 'I'

#__________________________________________________________________________

class Float(Variable):
    
    def __new__(cls, default=0.):
        
        return Variable.__new__(cls,'f',[float(default)])

    def __init__(self, default=0.):
        
        Variable.__init__(self)
        self.default = float(default)
    
    def set(self, value):
    
        self[0] = float(value)

    def type(self): return 'F'
