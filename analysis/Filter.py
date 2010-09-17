
class Filter(object):

    def __init__(self,verbose=False):
        
        self.verbose = verbose
        self.total = 0
        self.passing = 0
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        return "Filter %s\n"%(self.__class__.__name__)+\
               "Total: %i\n"%(self.total)+\
               "Pass:  %i"%(self.passing)

    def __add__(self,other):

        newfilter = self.__class__()
        newfilter.total = self.total + other.total
        newfilter.passing = self.passing + other.passing
        return newfilter
    
    def passes(self):

        if self.verbose: print "processing filter %s..."%(self.__class__.__name__)
        self.total += 1
