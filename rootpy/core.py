"""
This module contains base classes defining core funcionality
"""
import ROOT
import uuid
import inspect

def isbasictype(thing):
    """
    Is this thing a basic builtin numeric type?
    """
    return isinstance(thing, float) or \
           isinstance(thing, int) or \
           isinstance(thing, long)

def camelCaseMethods(cls):
    """
    A class decorator which adds camelCased methods
    which alias capitalized ROOT methods
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
        # Don't touch special methods and only consider capitalized methods
        if name[0] == '_' or name.islower():
            continue
        # Is this a method of the ROOT base class?
        if inspect.ismethod(member):
            # Make the first letter lowercase
            if len(name) == 1:
                new_name = name.lower()
            else:
                new_name = name[0].lower()+name[1:]
            # Is this method overridden in the child class?
            # If so, fix the method in the child
            try:
                submethod = getattr(cls, name)
                if not isinstance(submethod, ROOT.MethodProxy):
                    # The method was overridden
                    setattr(cls, new_name, submethod)
                    continue
            except AttributeError:
                pass
            setattr(root_base, new_name, member)
    return cls

class Object(object):
    """
    Overrides TObject methods. Name and title for TObject-derived classes are optional
    If no name is specified, a UUID is used to ensure uniqueness.
    """
    def __init__(self, name, title, *args, **kwargs):

        if name is None:
            name = uuid.uuid4().hex
        if title is None:
            title = ""
        self.__class__.__bases__[-1].__init__(self, name, title, *args, **kwargs)

    def Clone(self, name = None):

        if name is not None:
            clone = self.__class__.__bases__[-1].Clone(self, name)
        else:
            clone = self.__class__.__bases__[-1].Clone(self, uuid.uuid4().hex)
        clone.__class__ = self.__class__
        if hasattr(clone,"_post_init"):
            from .plotting.core import Plottable
            if isinstance(self, Plottable):
                #Plottable.__init__(clone)
                #clone.decorate(template_object = self)
                clone._post_init(**self.decorators())
            else:
                clone._post_init()
        return clone

    def __copy__(self):

        return self.Clone()

    def __deepcopy__(self, memo):

        return self.Clone()

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self.GetTitle())

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
