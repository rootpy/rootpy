import warnings

TYPES = {}

class register(object):

    def __init__(self, names=None, demote=None, builtin=False):

        self.names = names
        if names is not None:
            if type(names) not in (list, tuple):
                raise TypeError("names must be a list or tuple")
        self.demote = demote
        self.builtin = builtin

    def __call__(self, cls):
        
        init_methods = []
        
        if not self.builtin:
            if hasattr(cls, '_post_init'):
                init_methods.append(cls._post_init)

            # all rootpy classes which inherit from ROOT classes
            # must place the ROOT base class as the last class in the inheritance list
            rootbase = cls.__bases__[-1]
            cls_names = [rootbase.__name__]
        else:
            cls_names = [cls.__name__]
        
        if self.names is not None:
            cls_names += self.names
        
        cls_names_up = [name.upper() for name in cls_names]
        
        for name in cls_names_up:
            if name in TYPES:
                warnings.warn("Duplicate registration of class %s" % name)

            TYPES[name] = {
                'class': cls,
                'init': init_methods,
                'demote': self.demote
            }

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
    if cls_name in TYPES:
        entry = TYPES[cls_name]
        return entry['class'], entry['init']
    # ROOT class not registered...
    return None, []


def lookup_demotion(cls):

    cls_name = cls.__name__.upper()
    if cls_name in TYPES:
        entry = TYPES[cls_name]
        demote = entry['demote']
        if demote is None:
            return cls_name
        return demote
    return None
