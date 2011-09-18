import warnings

__TYPES = {}

def register(cls):
    
    init_methods = []
    if hasattr(cls, "_post_init"):
        init_methods.append(cls._post_init)

    rootbase = cls.__bases__[-1]
    if __TYPES.has_key(rootbase):
        warnings.warn("Overwriting previously registered class %s"% rootbase.__name__)

    __TYPES[rootbase] = {"class": cls, "init": init_methods}
    return cls

def lookup(cls):

    if __TYPES.has_key(cls):
        entry = __TYPES[cls]
        return entry['class'], entry['init']
    return cls, []
