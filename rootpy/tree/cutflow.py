import struct

class Cutflow(object):

    def __init__(self, names):
    
        self.__names = names
        self.__dict = dict((name, '0') for name in names)

    def __setitem__(self, item, value):

        self.__dict[item] = str(int(bool(value)))
    
    def bitstring(self):

        return ''.join([self.__dict[item] for item in self.__names])

    def int(self):

        return int(self.bitstring(), 2)
