from array import array

class Variable(object):
    
    def __init__(self): pass
        
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
#________________________________________________________________________

class Int(Variable, array):
    
    def __new__(cls, default=0):
        
        return array.__new__(cls,'i',[int(default)])

    def __init__(self, default=0):

        Variable.__init__(self)
        self.default = int(default)
    
    def set(self, value):
    
        self[0] = int(value)

    def __getitem__(self,i):
        
        return array.__getitem__(self,0)

    def __setitem__(self,i,value):

        array.__setitem__(self,0,int(value))

    def type(self): return 'I'

#__________________________________________________________________________

class Float(Variable, array):
    
    def __new__(cls, default=0.):
        
        return array.__new__(cls,'f',[float(default)])

    def __init__(self, default=0.):
        
        Variable.__init__(self)
        self.default = float(default)
    
    def set(self, value):
    
        self[0] = float(value)
    
    def __getitem__(self,i):

        return array.__getitem__(self,0)

    def __setitem__(self,i,value):

        array.__setitem__(self,0,float(value))

    def type(self): return 'F'
