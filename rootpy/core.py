"""
This module contains base classes defining core functionality
"""
import ROOT
import re
import uuid
import inspect
from . import rootpy_globals


class RequireFile(object):

    def __init__(self):

        if ROOT.gDirectory.GetName() == 'PyROOT':
            raise RuntimeError("You must first create a File "
                               "before creating a %s" % self.__class__.__name__)
        self.__directory = rootpy_globals.directory

    @staticmethod
    def cd(f):
        """
        A decorator
        Useful for TTree.Write...
        """
        def g(self, *args, **kwargs):
            pwd = rootpy_globals.directory
            self.__directory.cd()
            return f(self, *args, **kwargs)
            pwd.cd()
        return g


def wrap_call(cls, method, *args, **kwargs):
    """
    Will provide more detailed info in the case that
    a method call on a ROOT object raises a TypeError
    """
    pass


class _repr_mixin:

    def __str__(self):

        return self.__repr__()


class _copy_construct_mixin:

    def set_from(self, other):

        self.__class__.__bases__[-1].__init__(self, other)


class _resetable_mixin:

    def reset(self):

        self.__init__()


def isbasictype(thing):
    """
    Is this thing a basic builtin numeric type?
    """
    return isinstance(thing, (float, int, long))


FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(name):
    """
    http://stackoverflow.com/questions/1175208/
    elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', s1).lower()


def snake_case_methods(cls, debug=False):
    """
    A class decorator adding snake_case methods
    that alias capitalized ROOT methods
    """
    # Fix both the class and its corresponding ROOT base class
    root_base = cls.__bases__[-1]
    members = inspect.getmembers(root_base)
    # filter out any methods that already exist in lower and uppercase forms
    # i.e. TDirectory::cd and Cd...
    names = [item[0].capitalize() for item in members]
    duplicate_idx = set()
    seen = []
    for i, n in enumerate(names):
        try:
            idx = seen.index(n)
            duplicate_idx.add(i)
            duplicate_idx.add(idx)
        except ValueError:
            seen.append(n)
    for i, (name, member) in enumerate(members):
        if i in duplicate_idx:
            continue
        # Don't touch special methods or methods without cap letters
        if name[0] == '_' or name.islower():
            continue
        # Is this a method of the ROOT base class?
        if inspect.ismethod(member):
            # convert CamelCase to snake_case
            new_name = camel_to_snake(name)
            if debug:
                print "%s -> %s" % (name, new_name)
                if hasattr(cls, new_name):
                    raise ValueError(
                            '%s is already a method for %s' %
                            (new_name, cls.__name__))
            setattr(cls, new_name, getattr(cls, name))
    return cls


class Object(object):
    """
    Overrides TObject methods. Name and title for TObject-derived classes
    are optional. If no name is specified, a UUID is used to ensure uniqueness.
    """
    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__(
                self, name, title, *args, **kwargs)

    def Clone(self, name=None, title=None, **kwargs):

        if name is not None:
            clone = self.__class__.__bases__[-1].Clone(self, name)
        else:
            clone = self.__class__.__bases__[-1].Clone(self, uuid.uuid4().hex)
        clone.__class__ = self.__class__
        if title is not None:
            clone.SetTitle(title)
        if hasattr(clone, "_post_init"):
            from .plotting.core import Plottable
            if isinstance(self, Plottable):
                kwds = self.decorators
                kwds.update(kwargs)
                clone._post_init(**kwds)
            else:
                clone._post_init(**kwargs)
        return clone

    @property
    def name(self):

        return self.GetName()

    @name.setter
    def name(self, _name):

        self.SetName(_name)

    @property
    def title(self):

        return self.GetTitle()

    @title.setter
    def title(self, _title):

        self.SetTitle(_title)

    def __copy__(self):

        return self.Clone()

    def __deepcopy__(self, memo):

        return self.Clone()

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self.GetName())


class NamelessConstructorObject(Object):
    """
    Handle special cases like TGraph where the
    ROOT constructor does not take name/title
    """
    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__(self, *args, **kwargs)
        self.SetName(name)
        self.SetTitle(title)
