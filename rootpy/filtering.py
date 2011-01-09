
class Filter(object):

    def __init__(self):
        
        self.total = 0
        self.passing = 0
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        return "Filter %s\n"%(self.__class__.__name__)+\
               "Total: %i\n"%(self.total)+\
               "Pass:  %i"%(self.passing)

    def __add__(self, other):

        newfilter = self.__class__()
        newfilter.total = self.total + other.total
        newfilter.passing = self.passing + other.passing
        return newfilter
    
    def __call__(self, event):

        self.total += 1
        if self.passes():
            self.passing += 1
            return True
        return False

class FilterList(list):

    def __call__(self, event):

        for filter in self:
            if not filter(event):
                return False
