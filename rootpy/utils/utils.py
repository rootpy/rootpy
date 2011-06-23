import ROOT
from rootpy.core import Object, Plottable
from registry import lookup

def asrootpy(tobject):

    # is this object already converted?
    if isinstance(tobject, Object):
        return tobject
    template = Plottable()
    template.decorate(tobject)
    
    cls, inits = lookup(tobject.__class__)
    tobject.__class__ = cls
    for init in inits:
        init(tobject)
    
    if isinstance(tobject, Plottable):
        tobject.decorate(template)
    return tobject
