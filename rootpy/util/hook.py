import types

from .. import QROOT
from . import log; log = log[__name__]

# The below code is here for reference:
# How to hook anything you want..
"""
HOOKED_CLASSES = {}

TObject_meta = type(QROOT.TObject)

orig_meta_getattribute = TObject_meta.__getattribute__
def new_meta_getattribute(cls, name):
    #print cls, name
    if cls in HOOKED_CLASSES:
        hook = HOOKED_METHODS.get((cls, name), None)
        if hook:
            hook(orig_getattribute)
    return orig_meta_getattribute(cls, name)
TObject_meta.__getattribute__ = new_meta_getattribute

orig_getattribute = QROOT.TObject.__getattribute__
def new_getattribute(cls, name):
    x = orig_getattribute(cls, name)
    return x
QROOT.TObject.__getattribute__ = new_getattribute
"""

interesting = (types.FunctionType, types.MethodType, 
    property, staticmethod, classmethod)

class classhook(object):
    """
    Interpose the `hook` classes' methods onto the target `classes`.
    """

    def __init__(self, *classes):
        self.classes = classes

    def __call__(self, hook):
        self.hook = hook

        for cls in self.classes:
            # Attach a new class type with the original methods on it so that
            # super() works as expected.
            hookname = "_rootpy_{0}_OrigMethods".format(cls.__name__)
            newcls = types.ClassType(hookname, (), {})
            cls.__bases__ = (newcls,) + cls.__bases__

            for key, value in hook.__dict__.iteritems():
                if not isinstance(value, interesting):
                    continue
                orig_method = getattr(cls, key, None)
                if orig_method:
                    newcls.__dict__[key] = orig_method
                setattr(cls, key, value)

        return hook

class appendclass(object):
    """
    Append the methods/properties of `appender` onto `classes`. The methods
    being appended must not exist on any of the target classes.
    """
    def __init__(self, *classes):
        self.classes = classes

    def __call__(self, appender):
        for appendee in self.classes:
            for key, value in appender.__dict__.iteritems():
                if not isinstance(value, interesting):
                    continue
                assert not hasattr(appendee, key), (
                    "Don't override existing methods with appendclass")
                setattr(appendee, key, value)
                continue
        return appender
