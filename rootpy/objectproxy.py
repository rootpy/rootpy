# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import
import types


class MethodProxy(object):
    """
    Wrapper object for a method to be called.
    """
    def __init__(self, obj, func, name):

        self.obj, self.func, self.name = obj, func, name

    def __call__(self, *args, **kwds):

        return self.obj._method_call(self.name, self.func, *args, **kwds)


class ObjectProxy(object):

    __slots__ = ["_obj", "__weakref__"]

    def __init__(self, obj):

        object.__setattr__(self, "_obj", obj)

    def __nonzero__(self):

        return bool(object.__getattribute__(self, "_obj"))

    def __str__(self):

        return str(object.__getattribute__(self, "_obj"))

    def __repr__(self):

        return repr(object.__getattribute__(self, "_obj"))

    #
    # factories
    #
    _special_names = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__',
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__',
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__',
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__',
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__',
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__',
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
        '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__',
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__',
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__',
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__',
        '__truediv__', '__xor__', 'next',
    ]

    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""

        def make_method(name):
            def method(self, *args, **kw):
                return getattr(object.__getattribute__(self, "_obj"), name)(*args, **kw)
            return method

        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name) and not hasattr(cls, name):
                namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)

    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        ins = object.__new__(theclass)
        #theclass.__init__(ins, obj, *args, **kwargs)
        return ins

    def _method_call(self, ___name, ___func, *args, **kwds):
        """
        This method gets called before a method is called.
        """
        # pre-call hook for all calls.
        try:
            prefunc = getattr(self, '__pre__')
        except AttributeError:
            pass
        else:
            prefunc(___name, *args, **kwds)

        # pre-call hook for specific method.
        try:
            prefunc = getattr(self, '__pre__%s' % ___name)
        except AttributeError:
            pass
        else:
            prefunc(*args, **kwds)

        # get real method to call and call it
        rval = ___func(*args, **kwds)

        # post-call hook for specific method.
        try:
            postfunc = getattr(self, '__post__%s' % ___name, rval)
        except AttributeError:
            pass
        else:
            if type(postfunc) in [types.MethodType, types.FunctionType]:
                postfunc(*args, **kwds)

        # post-call hook for all calls.
        try:
            postfunc = getattr(self, '__post__', rval)
        except AttributeError:
            pass
        else:
            if type(postfunc) is [types.MethodType, types.FunctionType]:
                postfunc(___name, *args, **kwds)

        return rval

    def __setprehook__(self, name, func):

        setattr(self, "__pre__%s" % name, func)

    def __setposthook__(self, name, func):

        setattr(self, "__post__%s" % name, func)

    def __delattr__(self, name):

        delattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):

        if name in ['__setprehook__', '__setposthook__'] or \
                name.startswith('__post__') or name.startswith('__pre__'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_obj"), name, value)

    def __getattribute__(self, name):
        """
        Return a proxy wrapper object if this is a method call.
        """
        #if name.startswith('_'):
        #    return object.__getattribute__(self, name)
        #else:
        if name in ['__setprehook__', '__setposthook__', '_method_call'] \
                or name.startswith('__post__') or name.startswith('__pre__'):
            return object.__getattribute__(self, name)
        att = getattr(object.__getattribute__(self, "_obj"), name)
        if type(att) is types.MethodType:
            return MethodProxy(self, att, name)
        return att

    def __getitem__(self, key):
        """
        Delegate [] syntax.
        """
        name = '__getitem__'
        att = getattr(object.__getattribute__(self, "_obj"), name)
        pmeth = MethodProxy(self, att, name)
        return pmeth(key)

    def __setitem__(self, key, value):
        """
        Delegate [] syntax.
        """
        name = '__setitem__'
        att = getattr(object.__getattribute__(self, "_obj"), name)
        pmeth = MethodProxy(self, att, name)
        pmeth(key, value)
