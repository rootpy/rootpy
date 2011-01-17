import os
import re
import ROOT

class Cut(ROOT.TCut):
  
    def __init__(self, cut = ""):
        
        if type(cut) is file:
            cut = "".join(line.strip() for line in cut.readlines())
        elif isinstance(cut, Cut):
            cut = cut.GetTitle()
        ROOT.TCut.__init__(self, cut)
    
    def __and__(self, other):
        
        return Cut("(%s)&&(%s)"% (self, other))

    def __mul__(self, other):

        return self.__and__(other)
    
    def __or__(self, other):
        
        return Cut("(%s)||(%s)"% (self, other))
    
    def __add__(self, other):
        
        return self.__or__(other)

    def __neg__(self):

        if self:
            return Cut("!(%s)"% self)
        return Cut()

    def __pos__(self):
        
        return Cut(self)
    
    def __str__(self):
        
        return self.__repr__()
    
    def __repr__(self):
        
        return self.GetTitle()
         
    def __nonzero__(self):

        return str(self) != ''
    
    def safeString(self):
        
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
        return string

    def LaTeX(self):
        
        if not self:
            return ""
        string = str(self)
        string = string.replace("==", "=")
        string = string.replace("<=", "\leq")
        string = string.replace(">=", "\geq")
        string = string.replace("&&", " and ")
        string = string.replace("||", " or ")
        return string
