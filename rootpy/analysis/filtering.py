
class Filter(object):

    def __init__(self,buffer,verbose=False):
        
        self.buffer = buffer
        self.verbose = verbose
        self.total = 0
        self.passing = 0

    def __getstate__(self):

        return {"buffer":None,
                "verbose":False,
                "total":self.total,
                "passing":self.passing}
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        return "Filter %s\n"%(self.__class__.__name__)+\
               "Total: %i\n"%(self.total)+\
               "Pass:  %i"%(self.passing)

    def __add__(self,other):

        newfilter = self.__class__(self.buffer,self.verbose)
        newfilter.total = self.total + other.total
        newfilter.passing = self.passing + other.passing
        return newfilter
    
    def __nonzero__(self):

        if self.verbose: print "processing filter %s..."%(self.__class__.__name__)
        self.total += 1
        if self.passes():
            self.passing += 1
            return True
        return False
    
    def passes(self): pass

class FilterList(list):

    def __nonzero__(self):

        return all(self)
