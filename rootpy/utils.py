import ROOT
from .core import Object
from .plotting.core import Plottable
from .registry import lookup


def create(cls_name, *args, **kwargs):

    try:
        cls = getattr(ROOT, cls_name)
        obj = cls(*args, **kwargs)
        return asrootpy(obj)
    except:
        return None


def asrootpy(tobject, **kwargs):

    # is this object already converted?
    if isinstance(tobject, Object):
        return tobject

    template = Plottable()
    template.decorate(tobject)

    cls, inits = lookup(tobject.__class__)
    tobject.__class__ = cls
    for init in inits:
        init(tobject, **kwargs)

    if isinstance(tobject, Plottable):
        tobject.decorate(template)
    return tobject
