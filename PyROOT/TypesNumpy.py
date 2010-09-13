from numpy import *

class Variable:
    
    def __init__(self,dim):
        
        assert(dim>0)
        self.dim = dim

    def set(self,value):
    
        if self.dim == 1:
            self.val[0] = value
        else:
            for i in range(self.dim):
                self.val[i] = value[i]

    def reset(self):
        
        for i in range(self.dim):
            self.val[i] = self.default

    def value(self):
        
        if self.dim == 1:
            return self.val[0]
        else:
            return self.val
    
    def address(self):
        
        return self.val

    def dimension(self):

        return self.dim


class Bool(Variable):
    
    def __init__(self,default=0,dim=1):
        
        Variable.__init__(self,dim)
        self.default = bool(default)
        self.val = array([self.default for d in range(dim)], dtype=bool)
    
    def type(self): return 'B'


class Char(Variable):
    
    def __init__(self,default='a',dim=1):
        
        Variable.__init__(self,dim)
        self.default = str(default)[0]
        self.val = array([self.default for d in range(dim)], dtype=character)
    
    def type(self): return 'C'


class Int(Variable):
    
    def __init__(self,default=0,dim=1):
        
        Variable.__init__(self,dim)
        self.default = int(default)
        self.val = array([self.default for d in range(dim)], dtype=int32)
    
    def type(self): return 'I'


class UInt(Variable):
    
    def __init__(self,default=0,dim=1):
        
        Variable.__init__(self,dim)
        self.default = int(default)
        self.val = array([self.default for d in range(dim)], dtype=uint32)
    
    def type(self): return 'U'


class Float(Variable):
    
    def __init__(self,default=0.,dim=1):
        
        Variable.__init__(self,dim)
        self.default = float(default)
        self.val = array([self.default for d in range(dim)], dtype=float32)
    
    def type(self): return 'F'

class Double(Variable):

    def __init__(self,default=0.,dim=1):
        
        Variable.__init__(self,dim)
        self.default = long(default)
        self.val = array([self.default for d in range(dim)], dtype=float64)

    def type(self): return 'D'
