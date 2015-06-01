# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module contains base classes defining core functionality
"""
from __future__ import absolute_import

import ROOT
from .extern.shortuuid import uuid

__all__ = [
    'Object',
    'NamedObject',
    'NameOnlyObject',
    'NamelessConstructorObject',
]


class Object(object):
    """
    The rootpy-side base class of all ROOT subclasses in rootpy
    Classes that inherit from this class must also inherit from ROOT.TObject.
    """
    def Clone(self, name=None, title=None, shallow=False, **kwargs):
        if name is None:
            name = '{0}_{1}'.format(self.__class__.__name__, uuid())
        if shallow:
            # use the copy constructor
            clone = self._ROOT(self)
            clone.SetName(name)
        else:
            # a complete clone
            clone = super(Object, self).Clone(name)
        # cast
        clone.__class__ = self.__class__
        if title is not None:
            clone.SetTitle(title)
        if hasattr(clone, '_clone_post_init'):
            clone._clone_post_init(obj=self, **kwargs)
        elif hasattr(clone, '_post_init'):
            clone._post_init(**kwargs)
        return clone

    def copy_from(self, other):
        # not all classes implement Copy() correctly in ROOT, so use copy
        # constructor directly. Then again, not all classes in ROOT implement a
        # copy constructor or implement one correctly, so this might not work
        # everywhere...
        self._ROOT.__init__(self, other)

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
        return self.Clone(shallow=True)

    def __deepcopy__(self, memo):
        return self.Clone()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{0}('{1}')".format(
            self.__class__.__name__, self.GetName())

    # missing in PyROOT for Python 3
    __hash__ = object.__hash__


class NamedObject(Object):
    """
    Name and title for TNamed-derived classes are optional. If no name is
    specified, a UUID is used to ensure uniqueness.
    """
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        title = kwargs.pop('title', None)
        if name is None:
            name = '{0}_{1}'.format(self.__class__.__name__, uuid())
        if title is None:
            title = ''
        super(NamedObject, self).__init__(name, title, *args, **kwargs)


class NameOnlyObject(Object):
    """
    Handle special cases like TF1 where the constructor only takes a name.
    """
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        if name is None:
            name = '{0}_{1}'.format(self.__class__.__name__, uuid())
        super(NameOnlyObject, self).__init__(name, *args, **kwargs)


class NamelessConstructorObject(Object):
    """
    Handle special cases like TGraph where the ROOT constructor does not
    take name/title.
    """
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        title = kwargs.pop('title', None)
        if name is None:
            name = '{0}_{1}'.format(self.__class__.__name__, uuid())
        if title is None:
            title = ''
        super(NamelessConstructorObject, self).__init__(*args, **kwargs)
        self.SetName(name)
        self.SetTitle(title)
