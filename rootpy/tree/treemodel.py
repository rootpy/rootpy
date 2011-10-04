import types


class TreeModelMeta(type):
    
    def resolve_bases(cls, other):

        # union of bases of both classes
        bases = list(set(cls.__bases__).union(set(other.__bases__)))
        # sort bases so that subclassed bases are to the right of subclasses (for consistent MRO)
        bases.sort(cmp=lambda A, B: -1 if issubclass(A, B) else 1)
        return tuple(bases)
    
    def __add__(cls, other):

        attrs = dict(set(cls.get_attrs()).union(set(other.get_attrs())))
        return type('_'.join([cls.__name__, other.__name__]),
                    cls.resolve_bases(other), attrs)

    def __sub__(cls, other):
        
        attrs = dict(set(cls.get_attrs()).difference(set(other.get_attrs())))
        return type('_'.join([cls.__name__, other.__name__]),
                    cls.resolve_bases(other), attrs)
    
    def prefix(cls, name):

        attrs = dict([(name + attr, value) for attr, value in cls.get_attrs()])
        return type('_'.join([name, cls.__name__]),
                    cls.__bases__, attrs)

    def suffix(cls, name):
        
        attrs = dict([(attr + name, value) for attr, value in cls.get_attrs()])
        return type('_'.join([cls.__name__, name]),
                    cls.__bases__, attrs)


class TreeModel(object):

    __metaclass__ = TreeModelMeta

    @classmethod
    def get_attrs(cls):

        boring = dir(type('dummy', (object,), {})) + \
                 ['get_buffer', 'get_attrs', '__metaclass__']
        attrs = [item for item in inspect.getmembers(cls)
                if item[0] not in boring
                and not isinstance(item[1], types.FunctionType)
                and not isinstance(item[1], types.MethodType)]
        return attrs

    @classmethod
    def get_buffer(cls):
        
        buffer = TreeBuffer()
        for name, attr in cls.get_attrs():
            buffer[name] = attr()
        return buffer
