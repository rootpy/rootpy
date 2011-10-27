import os
import re
import ROOT
from .. import path


def cutop(func):
    
    def foo(self, other):
        other = Cut.convert(other)
        if not self:
            return other
        if not other:
            return self
        return func(self, other)
    return foo


def icutop(func):
    
    def foo(self, other):
        other = Cut.convert(other)
        if not self:
            self.SetTitle(other.GetTitle())
            return self
        if not other:
            return self
        return func(self, other)
    return foo


def _verbose(match):

    return '%s%s && %s%s' % \
           (match.group('left'),
            match.group('name'),
            match.group('name'),
            match.group('right'))

    
_TERNARY = re.compile('(?P<left>[<>=]+)\s*(?P<name>\w+)\s*(?P<right>[<>=]+)')


class Cut(ROOT.TCut):
    """
    A wrapper class around ROOT.TCut which implements logical operators
    """  
    def __init__(self, cut = "", from_file = False):
        
        if cut != "":
            if cut is None:
                cut = ""
            elif type(cut) is file:
                cut = "".join(line.strip() for line in cut.readlines())
            elif isinstance(cut, basestring) and from_file:
                ifile = open(path.expand(cut))
                cut = "".join(line.strip() for line in ifile.readlines())
                ifile.close()
            elif isinstance(cut, Cut):
                cut = cut.GetTitle()
        # TODO: add support for x < A < y etc.
        # convert to x < A && A < y before passing to __init__ below.
        
        cut = re.sub(_TERNARY, _verbose, cut)
        ROOT.TCut.__init__(self, cut)
    
    @staticmethod
    def convert(thing):

        if isinstance(thing, Cut):
            return thing
        elif isinstance(thing, basestring):
            return Cut(thing)
        elif thing is None:
            return Cut()
        try:
            cut = str(thing)
            return Cut(cut)
        except:
            raise TypeError("cannot convert %s to Cut"% type(thing))
    
    @cutop
    def __and__(self, other):
        """
        Return a new cut which is the logical AND of this cut and another
        """
        return Cut("(%s)&&(%s)"% (self, other))
    
    @cutop
    def __rand__(self, other):

        return self & other

    @cutop
    def __mul__(self, other):
        """
        Return a new cut which is the product of this cut and another
        """
        return Cut("(%s)*(%s)"% (self, other))
    
    @cutop
    def __rmul__(self, other):

        return self * other

    @icutop
    def __imul__(self, other):
        """
        Multiply other cut with self and return self
        """
        self.SetTitle("(%s)*(%s)"% (self, other))
        return self

    @cutop
    def __or__(self, other):
        """
        Return a new cut which is the logical OR of this cut and another
        """
        return Cut("(%s)||(%s)"% (self, other))
    
    @cutop
    def __ror__(self, other):

        return self | other

    @cutop
    def __add__(self, other):
        """
        Return a new cut which is the sum of this cut and another
        """
        return Cut("(%s)+(%s)"% (self, other))

    @cutop
    def __radd__(self, other):

        return self + other
    
    @icutop
    def __iadd__(self, other):
        """
        Add other cut to self and return self
        """
        self.SetTitle("(%s)+(%s)"% (self, other))
        return self
    
    @cutop
    def __sub__(self, other):
        """
        Return a new cut which is the difference of this cut and another
        """
        return Cut("(%s)-(%s)"% (self, other))

    @cutop
    def __rsub__(self, other):

        return self - other
    
    @icutop
    def __isub__(self, other):
        """
        Subtract other cut to self and return self
        """
        self.SetTitle("(%s)-(%s)"% (self, other))
        return self

    def __neg__(self):
        """
        Return a new cut which is the negation of this cut
        """
        if not self:
            return Cut()
        return Cut("!(%s)"% self)

    def __pos__(self):
        
        return Cut(self)
    
    def __str__(self):
        
        return self.__repr__()
    
    def __repr__(self):
        
        return self.GetTitle()
         
    def __nonzero__(self):
        """
        A cut evaluates to False if it is empty (null cut).
        This has no affect on its actual boolean value within the context of
        a ROOT.TTree selection.
        """
        return str(self) != ''
    
    def safe(self):
        """
        Returns a string representation with special characters
        replaced by safer characters for use in filenames for example.
        """
        if not self:
            return ""
        string = str(self)
        string = string.replace("==", "-eq-")
        string = string.replace("<=", "-leq-")
        string = string.replace(">=", "-geq-")
        string = string.replace("<", "-lt-")
        string = string.replace(">", "-gt-")
        string = string.replace("&&", "-and-")
        string = string.replace("||", "-or-")
        string = string.replace("(", "L")
        string = string.replace(")", "R")
        string = string.replace(" ", "")
        return string

    def latex(self):
        """
        Returns a string representation for use in LaTeX
        """ 
        if not self:
            return ""
        string = str(self)
        string = string.replace("==", "=")
        string = string.replace("<=", "\leq")
        string = string.replace(">=", "\geq")
        string = string.replace("&&", " and ")
        string = string.replace("||", " or ")
        return string

    
    def where(self):

        string = str(self)
        string = string.replace('&&','&')
        string = string.replace('||','|')
        return string
    
    def replace(self, name, newname):
        
        if not re.match("[a-zA-Z]\w*", name):
            return None
        if not re.match("[a-zA-Z]\w*", newname):
            return None
        
        def _replace(match):
            
            return match.group(0).replace(match.group('name'), newname)
        
        pattern = re.compile("(\W|^)(?P<name>"+name+")(\W|$)")
        cut = re.sub(pattern, _replace, str(self))
        return Cut(cut)
