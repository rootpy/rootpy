import warnings

__TYPES = {}

def register(cls):
    
    init_methods = []
    if hasattr(cls, "_post_init"):
        init_methods.append(cls._post_init)

    # all rootpy classes which inherit from ROOT classes
    # must place the ROOT base class as the last class in the inheritance list
    rootbase = cls.__bases__[-1]
    cls_name = rootbase.__name__.upper()
    if cls_name in __TYPES:
        warnings.warn("Overwriting previously registered class %s"% rootbase.__name__)

    __TYPES[cls_name] = {"class": cls, "init": init_methods}
    return cls

def lookup(cls):

    cls_name = cls.__name__.upper()
    rootpy_cls, inits = lookup_by_name(cls.__name__)
    if rootpy_cls is None:
        # ROOT class not registered, pass-through
        return cls, []
    return rootpy_cls, inits

def lookup_by_name(cls_name):

    cls_name = cls_name.upper()
    if cls_name in __TYPES:
        entry = __TYPES[cls_name]
        return entry['class'], entry['init']
    # ROOT class not registered...
    return None, []
