# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import types
import sys

from .inject_closure import inject_closure_values
from . import log; log = log[__name__]

__all__ = [
    'super_overridden',
    'uses_super',
    'classhook',
    'appendclass',
]

# The below code is here for reference:
# How to hook anything you want..
# TODO(pwaller): Delete this if no-one needs it after a month or two.
"""
from .. import QROOT

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

INTERESTING = (
    types.FunctionType, types.MethodType,
    property, staticmethod, classmethod)


def super_overridden(cls):
    """
    This decorator just serves as a reminder that the super function behaves
    differently. It doesn't actually do anything, that happens inside
    ``classhook.hook_class``.
    """
    cls.__rootpy_have_super_overridden = True
    return cls


def uses_super(func):
    """
    Check if the function/property/classmethod/staticmethod uses the `super` builtin
    """
    if isinstance(func, property):
        return any(uses_super(f) for f in (func.fget, func.fset, func.fdel) if f)
    elif isinstance(func, (staticmethod, classmethod)):
        if sys.version_info >= (2, 7):
            func = func.__func__
        elif isinstance(func, staticmethod):
            func = func.__get__(True)
        else: # classmethod
            func = func.__get__(True).im_func
    if sys.version_info[0] >= 3:
        return 'super' in func.__code__.co_names
    return 'super' in func.func_code.co_names


class classhook(object):
    """
    Interpose the `hook` classes' methods onto the target `classes`.

    Note, it is also necessary to decorate these classes with @super_overridden
    to indicate at the usage site that the super method may behave differently
    than you expect.

    The trick is that we want the hook function to call `super(ClassBeingHooked, self)`,
    but there are potentially multiple ClassesBeingHooked. Therefore, instead
    you must write `super(MyHookClass, self)` and the super method is replaced
    at hook-time through bytecode modification with another one which does the
    right thing.

    Example usage:

    @classhook(ROOT.TH1)
    @super_overridden
    class ChangeBehaviour(object):
        def Draw(self, *args):
            # Call the original draw function
            result = super(ChangeBehaviour, self).Draw(*args)
            # do something with the result here
            return result
    """
    def overridden_super(self, target, realclass):
        class rootpy_overridden_super(super):
            def __init__(self, cls, *args):
                if cls is target:
                    cls = realclass
                super(rootpy_overridden_super, self).__init__(cls, *args)
        return rootpy_overridden_super

    def __init__(self, *classes):
        self.classes = classes

    def hook_class(self, cls, hook):
        # Attach a new class type with the original methods on it so that
        # super() works as expected.
        hookname = "_rootpy_{0}_OrigMethods".format(cls.__name__)
        newcls = type(hookname, (), {})
        cls.__bases__ = (newcls,) + cls.__bases__

        # For every function-like (or property), replace `cls`'s methods
        for key, value in hook.__dict__.items():
            if not isinstance(value, INTERESTING):
                continue

            # Save the original methods onto the newcls which has been
            # injected onto our bases, so that the originals can be called with
            # super().
            orig_method = getattr(cls, key, None)
            if orig_method:
                setattr(newcls, key, orig_method)
                #newcls.__dict__[key] = orig_method

            newmeth = value
            if uses_super(newmeth):
                assert getattr(hook, "__rootpy_have_super_overridden", None), (
                    "Hook class {0} is not decorated with @super_overridden! "
                    "See the ``hook`` module to understand why this must be "
                    "the case for all classes overridden with @classhook"
                    .format(hook))
                # Make super behave as though the class hierarchy is what we'd
                # like.
                newsuper = self.overridden_super(hook, cls)
                newmeth = inject_closure_values(value, super=newsuper)
            setattr(cls, key, newmeth)

    def __call__(self, hook):
        """
        Hook the decorated class onto all `classes`.
        """
        for cls in self.classes:
            self.hook_class(cls, hook)
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
            for key, value in appender.__dict__.items():
                if not isinstance(value, INTERESTING):
                    continue
                assert not hasattr(appendee, key), (
                    "Don't override existing methods with appendclass")
                assert not uses_super(value), ("Don't use the super class with "
                    "@appendclass, use @classhook instead")
                setattr(appendee, key, value)
                continue
        return appender
