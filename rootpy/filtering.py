try:
    from prettytable import PrettyTable
except: pass

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
        self.details = {}
    
    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        return "Filter %s\n"%(self.__class__.__name__)+\
               "Total: %i\n"%(self.total)+\
               "Pass:  %i"%(self.passing)

    def __add__(self, other):
        
        if other.__class__ != self.__class__:
            raise TypeError("Filters must be of the same clas when adding them")
        newfilter = self.__class__()
        newfilter.total = self.total + other.total
        newfilter.passing = self.passing + other.passing
        newfilter.details = dict([(detail, self.details[detail]+other.details[detail]) for detail in self.details.keys()])
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
    @staticmethod
    def merge(list1, list2):
        
        filterlist = FilterList()
        for f1,f2 in zip(list1,list2):
            if f1.__class__ != f2.__class__:
                raise TypeError("incompatible FilterLists")
            filterlist.append(f1+f2)
        return filterlist
    
    def __setitem__(self, filter):

        if not isinstance(filter, Filter):
            raise TypeError("FilterList can only hold objects inheriting from Filter")
        super(FilterList, self).__setitem__(filter)
    
    def append(self, filter):
        
        if not isinstance(filter, Filter):
            raise TypeError("FilterList can only hold objects inheriting from Filter")
        super(FilterList, self).append(filter)

    def __str__(self):

        return self.__repr__()
    
    def __repr__(self):

        if len(self) > 0:
            table = PrettyTable(["Filter", "Pass"])
            table.set_field_align("Filter","l")
            table.set_field_align("Pass","l")
            table.add_row(["Total", self[0].total])
            for filter in self:
                table.add_row([filter.__class__.__name__, filter.passing])
            return str(table)
        return "Empty FilterList"
    
    def __call__(self, event):

        for filter in self:
            if not filter(event):
                return False
        return True
