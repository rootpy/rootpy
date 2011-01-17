import warnings

__TYPES = {}

def register(cls, init_methods = None):
    
    if init_methods is not None:
        if type(init_methods) is not list:
            init_methods = [init_methods]
    else:
        init_methods = []

    rootbase = cls.__bases__[-1]
    if __TYPES.has_key(rootbase):
        warnings.warn("Overwriting previously registered class %s"% rootbase.__name__)

    __TYPES[rootbase] = {"class": cls, "init": init_methods}

def lookup(cls):

    if __TYPES.has_key(cls):
        entry = __TYPES[cls]
        return entry['class'], entry['init']
    return cls, []
