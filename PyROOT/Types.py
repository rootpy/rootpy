from array import array

class Variable(object):
    
    def __init__(self): pass
        
    def reset(self):
        
        for i in range(self.__len__()):
            self[i] = self.default

    def value(self):
        
        if self.__len__() == 1:
            return self[0]
        else:
            return self
    
    def address(self):
        
        return self

    def __str__(self):
        
        return self.__repr__()

    def __repr__(self):

        if self.__len__()==1:
            return "%s(%s)"%(self.__class__.__name__,self[0])
        else:
            return "%s(%s)"%(self.__class__.__name__,",".join("%s"%val for val in self))

#________________________________________________________________________

class Int(Variable, array):
    
    def __new__(cls, default=0, dim=1):
        
        return array.__new__(cls,'i',[int(default) for d in xrange(dim)])

    def __init__(self, default=0, dim=1):

        Variable.__init__(self)
        self.default = int(default)
    
    def set(self, value):
    
        if self.__len__() == 1:
            self[0] = int(value)
        else:
            for i in range(self.__len__()):
                self[i] = int(value[i])

    def type(self): return 'I'

#__________________________________________________________________________

class Float(Variable, array):
    
    def __new__(cls, default=0., dim=1):
        
        return array.__new__(cls,'f',[float(default) for d in xrange(dim)])

    def __init__(self, default=0., dim=1):
        
        Variable.__init__(self)
        self.default = float(default)
    
    def set(self, value):
    
        if self.__len__() == 1:
            self[0] = float(value)
        else:
            for i in range(self.__len__()):
                self[i] = float(value[i])

    def type(self): return 'F'

#__________________________________________________________________________

class Double(Variable, array):
    
    def __new__(cls, default=0., dim=1):
        
        return array.__new__(cls,'d',[float(default) for d in xrange(dim)])

    def __init__(self, default=0., dim=1):
        
        Variable.__init__(self)
        self.default = float(default)
    
    def set(self, value):
    
        if self.__len__() == 1:
            self[0] = float(value)
        else:
            for i in range(self.__len__()):
                self[i] = float(value[i])

    def type(self): return 'F'
