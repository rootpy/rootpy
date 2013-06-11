# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import inspect
from cStringIO import StringIO
import types

import ROOT

from .. import log; log = log["__name__"]

from .treetypes import Column
from .treebuffer import TreeBuffer


class TreeModelMeta(type):
    """
    Metaclass for all TreeModels
    Addition/subtraction of TreeModels is handled
    as set union and difference of class attributes
    """
    def __new__(cls, name, bases, dct):

        for attr, value in dct.items():
            TreeModelMeta.checkattr(attr, value)
        return type.__new__(cls, name, bases, dct)

    def __add__(cls, other):

        return type('_'.join([cls.__name__, other.__name__]),
                    (cls, other), {})

    def __iadd__(cls, other):

        return cls.__add__(other)

    def __sub__(cls, other):

        attrs = dict(set(cls.get_attrs()).difference(set(other.get_attrs())))
        return type('_'.join([cls.__name__, other.__name__]),
                    (TreeModel,), attrs)

    def __isub__(cls, other):

        return cls.__sub__(other)

    def __setattr__(cls, attr, value):

        TreeModelMeta.checkattr(attr, value)
        type.__setattr__(cls, attr, value)

    @classmethod
    def checkattr(metacls, attr, value):
        """
        Only allow class attributes that are instances of
        rootpy.types.Column, ROOT.TObject, or ROOT.ObjectProxy
        """
        if not isinstance(value, (types.MethodType,
                                  types.FunctionType,
                                  classmethod,
                                  staticmethod,
                                  property)):
            if attr in dir(type('dummy', (object,), {})) + \
                    ['__metaclass__']:
                return
            if attr.startswith('_'):
                raise SyntaxError("TreeModel attribute ``%s`` "
                                  "must not start with ``_``" % attr)
            if not inspect.isclass(value):
                if not isinstance(value, Column):
                    raise TypeError("TreeModel attribute ``%s`` "
                                    "must be an instance of "
                                    "``rootpy.types.Column``" % attr)
                return
            if not issubclass(value, (ROOT.TObject, ROOT.ObjectProxy)):
                raise TypeError("TreeModel attribute ``%s`` must inherit "
                                "from ``ROOT.TObject`` or ``ROOT.ObjectProxy``"
                                % attr)

    def prefix(cls, name):
        """
        Create a new TreeModel where class attribute
        names are prefixed with ``name``
        """
        attrs = dict([(name + attr, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([name, cls.__name__]),
                    (TreeModel,), attrs)

    def suffix(cls, name):
        """
        Create a new TreeModel where class attribute
        names are suffixed with ``name``
        """
        attrs = dict([(attr + name, value) for attr, value in cls.get_attrs()])
        return TreeModelMeta('_'.join([cls.__name__, name]),
                    (TreeModel,), attrs)

    def get_attrs(cls):
        """
        Get all class attributes
        """
        ignore = dir(type('dummy', (object,), {})) + \
                 ['__metaclass__']
        attrs = [item for item in inspect.getmembers(cls)
                 if item[0] not in ignore
                 and not isinstance(item[1],
                     (types.FunctionType,
                      types.MethodType,
                      classmethod,
                      staticmethod,
                      property))]
        return attrs

    def to_struct(cls, name=None):
        """
        Convert the TreeModel into a compiled C struct
        """
        if name is None:
            name = cls.__name__
        basic_attrs = dict([(attr_name, value)
                            for attr_name, value in cls.get_attrs()
                            if isinstance(value, Column)])
        if not basic_attrs:
            return None
        src = 'struct %s {' % name
        for attr_name, value in basic_attrs.items():
            src += '%s %s;' % (value.type.typename, attr_name)
        src += '};'
        if ROOT.gROOT.ProcessLine(src) != 0:
            return None
        return getattr(ROOT, name, None)

    def __repr__(cls):

        out = StringIO()
        for name, value in cls.get_attrs():
            print >> out, '%s -> %s' % (name, value)
        return out.getvalue()[:-1]

    def __str__(cls):

        return repr(cls)


class TreeModel(object):

    __metaclass__ = TreeModelMeta

    def __new__(cls):
        """
        Return a TreeBuffer for this TreeModel
        """
        treebuffer = TreeBuffer()
        for name, attr in cls.get_attrs():
            treebuffer[name] = attr()
        return treebuffer
