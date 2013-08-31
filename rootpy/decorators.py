# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os
import re
import inspect
import warnings

from .context import preserve_current_directory
from .extern import decorator
from . import ROOT, ROOT_VERSION


CONVERT_SNAKE_CASE = os.getenv('NO_ROOTPY_SNAKE_CASE', False) is False


def requires_ROOT(version, exception=False):
    """
    A decorator for functions or methods that require a minimum ROOT version.
    If `exception` is False (the default) a warning is issued and None is
    returned, otherwise a `NotImplementedError` exception is raised.
    `exception` may also be an `Exception` in which case it will be raised
    instead of `NotImplementedError`.
    """
    @decorator.decorator
    def wrap(f, *args, **kwargs):
        if ROOT_VERSION < version:
            msg = ("{0} requires at least ROOT {1} "
                   "but you are using {2}".format(
                       f.__name__, version, ROOT_VERSION))
            if inspect.isclass(exception) and issubclass(exception, Exception):
                raise exception
            elif exception:
                raise NotImplementedError(msg)
            warnings.warn(msg)
            return None
        return f(*args, **kwargs)
    return wrap


def _get_qualified_name(thing):

    if inspect.ismodule(thing):
        return thing.__file__
    if inspect.isclass(thing):
        return '{0}.{1}'.format(thing.__module__, thing.__name__)
    if inspect.ismethod(thing):
        return '{0}.{1}'.format(thing.im_class.__name__, thing.__name__)
    if inspect.isfunction(thing):
        return thing.__name__
    return repr(thing)


@decorator.decorator
def method_file_check(f, self, *args, **kwargs):
    """
    A decorator to check that a TFile as been created before f is called.
    This function can decorate methods.
    """
    # This requires special treatment since in Python 3 unbound methods are
    # just functions: http://stackoverflow.com/a/3589335/1002176 but to get
    # consistent access to the class in both 2.x and 3.x, we need self.
    curr_dir = ROOT.gDirectory.func()
    if isinstance(curr_dir, ROOT.TROOT):
        raise RuntimeError(
            "You must first create a File before calling {0}.{1}".format(
                self.__class__.__name__, _get_qualified_name(f)))
    if not curr_dir.IsWritable():
        raise RuntimeError(
            "Calling {0}.{1} requires that the "
            "current File is writable".format(
                self.__class__.__name__, _get_qualified_name(f)))
    return f(self, *args, **kwargs)


@decorator.decorator
def method_file_cd(f, self, *args, **kwargs):
    """
    A decorator to cd back to the original directory where this object was
    created (useful for any calls to TObject.Write).
    This function can decorate methods.
    """
    with preserve_current_directory():
        self.GetDirectory().cd()
        return f(self, *args, **kwargs)


@decorator.decorator
def chainable(f, self, *args, **kwargs):
    """
    Decorator which causes a 'void' function to return self

    Allows chaining of multiple modifier class methods.
    """
    # perform action
    f(self, *args, **kwargs)
    # return reference to class.
    return self


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
    if not CONVERT_SNAKE_CASE:
        return cls
    # Fix both the class and its corresponding ROOT base class
    #TODO use the class property on Object
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
        if inspect.ismethod(member) or inspect.isfunction(member):
            # convert CamelCase to snake_case
            new_name = camel_to_snake(name)
            if debug:
                print "{0} -> {1}".format(name, new_name)
                if hasattr(cls, new_name):
                    raise ValueError(
                        '`{0}` is already a method for `{1}`'.format(
                        new_name, cls.__name__))

            # Use a __dict__ lookup rather than getattr because we _want_ to
            # obtain the _descriptor_, and not what the descriptor gives us
            # when it is `getattr`'d.
            value = None
            for c in cls.mro():
                if name in c.__dict__:
                    value = c.__dict__[name]
                    break
            # <neo>Woah, a use for for-else</neo>
            else:
                # Weird. Maybe the item lives somewhere else, such as on the
                # metaclass?
                value = getattr(cls, name)

            setattr(cls, new_name, value)
    return cls
