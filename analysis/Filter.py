
class Filter(object):

    def __init__(self,verbose=False):
        
        self.verbose = verbose
        self.total = 0
        self.passing = 0
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        return "Filter %s\n\
                Total: %i\n\
                Pass:  %i"%(self.__class__.__name__,self.total,self.passing)
    
    def passes(self):

        if self.verbose: print "processing filter %s..."%(self.__class__.__name__)
        self.total += 1
