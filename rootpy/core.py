"""
This module contains base classes defining core funcionality
"""
import ROOT
import uuid

def isbasictype(thing):

    return isinstance(thing, float) or \
           isinstance(thing, int) or \
           isinstance(thing, long)

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
        self.__class__.__bases__[-1].__init__\
            (self, name, title, *args, **kwargs)

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

        return "%s(%s)"%(self.__class__.__name__, self.GetTitle())
    
    def __getattr__(self, attr):

        try:
            return super(Object, self).__getattr__(attr)
        except AttributeError, e:
            try:
                return super(Object, self).__getattr__(attr.capitalize())
            except:
                raise e

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
