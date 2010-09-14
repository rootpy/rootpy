
class Filter(object):

    def __init__(self,verbose=False):
        
        self.verbose = verbose

    def passes(self):

        if self.verbose: print "processing filter %s..."%(self.__class__.__name__)
