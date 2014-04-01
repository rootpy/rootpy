import sys

from inspect import getfile
from types import ModuleType

from .. import log; log = log[__name__]

log.show_stack(limit=2)


class computed_once_classproperty(property):
    """
    A property whose value is computed exactly once, then saved onto the target
    class.
    """
    def __get__(self, object_, type_=None):
        result = super(computed_once_classproperty, self).__get__(object_, type_)
        propname = self.fget.__name__
        # Remove the property from the class itself
        setattr(type_, propname, result)
        return result


class ModuleFacade(object):
    def __repr__(self):
        orig = super(ModuleFacade, self).__repr__()
        return "{0}({1})".format(type(self).__name__, orig)


class Facade(object):

    def __init__(self, name, **kwargs):
        """
        Use kwargs to force user to write them out for explicitness.
        """
        self.name = name
        _, _, self.name_lastpart = name.rpartition(".")
        self.expose_internal = kwargs.pop("expose_internal", True)
        self.submodule = kwargs.pop("submodule", False)

    def __call__(self, cls):
        """
        Decorate `cls`
        """
        expose_internal = self.expose_internal

        if self.submodule:
            self.name += "." + cls.__name__

        if self.name not in sys.modules:
            orig = ModuleType(self.name)
            orig.__name__ = self.name
            orig.__file__ = getfile(cls)
        else:
            orig = sys.modules[self.name]

        if isinstance(orig, ModuleFacade):
            raise TypeError("Facade() used inside module which is already "
                              "wrapped - only once Facade() allowed per module."
                              " inside {0}".format(orig))

        class _wrapper_cls(cls, ModuleFacade, ModuleType, object):
            _facade_wrapped = orig
            _facade_cls = cls

            def __dir__(self):
                items = set()
                items.update(self.__dict__)
                items.update(self._facade_cls.__dict__)

                if hasattr(self._facade_cls, "__dir__"):
                    items.update(self._facade_cls.__dir__(self))

                if expose_internal:
                    items.update(orig.__dict__)

                return sorted(items)

            def __getattr__(self, key):
                if expose_internal and hasattr(orig, key):
                    return getattr(orig, key)
                sup = super(_wrapper_cls, self)
                if hasattr(sup, "__getattr__"):
                    result = sup.__getattr__(key)
                    if result is not None:
                        return result
                raise AttributeError("'{0}' object has no attribute '{1}'"
                    .format(self, key))

        _wrapper_cls.__name__ = "ModuleFacade({0})".format(cls.__name__)
        inst = _wrapper_cls(self.name)
        sys.modules[self.name] = inst

        for key in "__name__ __doc__ __file__ __path__".split():
            if hasattr(orig, key):
                setattr(inst, key, getattr(orig, key))

        return inst
