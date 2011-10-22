

class Cutflow(object):

    def __init__(self, names):
    
        self.__names = names
        self.__dict = None
        self.reset()

    def __setitem__(self, name, passes):

        self.__dict[name] = str(int(bool(passes)))
    
    def passes(self, name):
        
        self.__dict[name] = '1'
    
    def stages(self):
        
        self.reset()
        yield self
        for name in self.__names:
           self.passes(name)
           yield self
        self.reset()

    def reset(self):

        self.__dict = dict((name, '0') for name in self.__names)

    def bitstring(self):

        return ''.join([self.__dict[item] for item in self.__names])

    def int(self):

        return int(self.bitstring(), 2)
