"""
This module defines a framework for filtering Trees.
The user must write a class which inherits from Filter and
"""
class Filter(object):
    """
    The base class from which all filter classes must inherit from.
    The derived class must override the passes method which returns True
    if ths event passes and returns False if not.
    The number of passing and failing events are recorded and may be used
    later to create a cut-flow.
    """
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
        if self.passes(event):
            self.passing += 1
            return True
        return False

    def passes(self, event):

        raise NotImplementedError("You must override this method in your derived class")

class FilterList(list):
    """
    Creates a list of Filters for convenient evaluation of a sequence of Filters.
    """
    def __call__(self, event):

        for filter in self:
            if not filter(event):
                return False
