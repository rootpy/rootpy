import ROOT
from .core import Object
from .plotting.core import Plottable
from .registry import lookup

def create(cls_name):

    try:
        exec 'obj = ROOT.%s()' % cls_name
        return asrootpy(obj)
    except:
        return None
    
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
