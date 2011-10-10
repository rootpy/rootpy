import warnings

TYPES = {}

class register(object):

    def __init__(self, shortcode=None, typename=None, demote=None, builtin=False):

        self.shortcode = shortcode
        self.typename = typename
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
            cls_name = rootbase.__name__
        elif self.typename is not None:
            cls_name = self.typename
        else:
            cls_name = cls.__name__
        
        cls_name_up = cls_name.upper()
        
        if cls_name_up in TYPES:
            warnings.warn("Duplicate registration of class %s" % cls_name)

        TYPES[cls_name_up] = {
            'class': cls,
            'init': init_methods,
            'demote': self.demote
        }

        if self.shortcode is not None:
            shortcode_up = self.shortcode.upper()
            if shortcode_up in TYPES:
                warnings.warn("Duplicate registration of type %s" % self.shortcode)

            TYPES[shortcode_up] = {
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
            return cls.__name__
        return demote
    return None
