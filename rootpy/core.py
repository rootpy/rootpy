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
    root_base = cls.__bases__[-1]
    method_names = dir(root_base)
    for method_name in method_names:
        if not method_name.startswith('_') and \
               method_name[0].isupper():
            if inspect.ismethod(getattr(root_base, method_name)):
                if len(method_name) == 1:
                    setattr(cls, method_name.lower(), getattr(root_base, method_name))
                else:
                    setattr(cls, method_name[0].lower()+method_name[1:], getattr(root_base, method_name))
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
        self.root_base = self.__class__.__bases__[-1]
        self.root_base.__init__(self, name, title, *args, **kwargs)

    def Clone(self, name = None):

        if name is not None:
            clone = self.root_base.Clone(self, name)
        else:
            clone = self.root_base.Clone(self, uuid.uuid4().hex)
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

        return "%s(%s)"%(self.__class__.__name__, self.GetTitle())
    

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
