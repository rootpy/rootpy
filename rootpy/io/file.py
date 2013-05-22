# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module enhances IO-related ROOT functionality
"""
import ROOT

from ..core import Object
from ..decorators import snake_case_methods
from .. import asrootpy, QROOT
from . import utils, DoesNotExist
from ..util import path

import tempfile
import os
import warnings


__all__ = [
    'Directory',
    'File',
    'TemporaryFile',
    'root_open',
    'open', # deprecated
]


VALIDPATH = '^(?P<file>.+.root)(?:[/](?P<path>.+))?$'
GLOBALS = {}


def wrap_path_handling(f):

    def get(self, name, **kwargs):

        _name = os.path.normpath(name)
        if _name == '.':
            return self
        if _name == '..':
            return self._parent
        try:
            dir, _, path = _name.partition(os.path.sep)
            if path:
                if dir == '..':
                    return self._parent.Get(path, **kwargs)
                else:
                    _dir = f(self, dir)
                    if not isinstance(_dir, _DirectoryBase):
                        raise DoesNotExist
                    _dir._parent = self
                    _dir._path = os.path.join(self._path, dir)
                    thing = _dir.Get(path, **kwargs)
            else:
                thing = f(self, _name, **kwargs)
                if isinstance(thing, _DirectoryBase):
                    thing._parent = self
            if isinstance(thing, _DirectoryBase):
                if isinstance(self, File):
                    thing._path = os.path.normpath(
                            (':' + os.path.sep).join([self._path, _name]))
                else:
                    thing._path = os.path.normpath(
                            os.path.join(self._path, _name))
            return thing
        except DoesNotExist:
            raise DoesNotExist("requested path '%s' does not exist in %s" %
                    (name, self._path))
    return get


class _DirectoryBase(Object):
    """
    A mixin (can't stand alone).
    """
    def walk(self, top=None, class_pattern=None):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top, class_pattern=class_pattern)

    def __getattr__(self, attr):
        """
        Natural naming support. Now you can get an object from a
        File/Directory with::

            myfile.somedir.otherdir.histname
        """
        # Be careful! If ``__getattr__`` ends up being called again here,
        # this can end up in an "infinite" recursion and stack overflow.

        # Directly call ROOT's Get() here since ``attr`` must anyway be a valid
        # identifier (not a path including subdirectories).
        thing = super(_DirectoryBase, self).Get(attr)
        if not thing:
            raise AttributeError
        thing = asrootpy(thing)
        if isinstance(thing, Directory):
            thing._path = os.path.join(self._path, thing.GetName())
            thing._parent = self
        return thing

    def __getitem__(self, name):

        return self.Get(name)

    def __iter__(self):

        return self.walk()

    def keys(self):

        return self.GetListOfKeys()

    def unique_keys(self):

        keys = {}
        for key in self.keys():
            keys[key.GetName()] = key
        return keys.values()

    @wrap_path_handling
    def Get(self, name, **kwargs):
        """
        Attempt to convert requested object into rootpy form
        """
        thing = super(_DirectoryBase, self).Get(name)
        if not thing:
            raise DoesNotExist
        return asrootpy(thing, **kwargs)

    def GetRaw(self, name):
        """
        Raw access without conversion into rootpy form
        """
        thing = super(_DirectoryBase, self).Get(name)
        if not thing:
            raise DoesNotExist
        return thing

    @wrap_path_handling
    def GetDirectory(self, name, **kwargs):
        """
        Return a Directory object rather than TDirectory
        """
        rdir = super(_DirectoryBase, self).GetDirectory(name)
        if not rdir:
            raise DoesNotExist
        return asrootpy(rdir, **kwargs)


@snake_case_methods
class Directory(_DirectoryBase, QROOT.TDirectoryFile):
    """
    Inherits from TDirectory
    """
    def __init__(self, name, title, *args, **kwargs):

        super(Directory, self).__init__(name, title, *args, **kwargs)
        self._post_init()

    def _post_init(self):

        self._path = self.GetName()
        self._parent = ROOT.gDirectory.func()

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()


@snake_case_methods
class File(_DirectoryBase, QROOT.TFile):
    """
    Wrapper for TFile that adds various convenience functions.

    >>> from rootpy.test import filename
    >>> f = File(filename, 'read')

    """
    def __init__(self, name, *args, **kwargs):

        # trigger finalSetup
        ROOT.kTRUE
        super(File, self).__init__(name, *args, **kwargs)
        self._path = self.GetName()
        self._parent = self

    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):

        self.Close()
        return False

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()


@snake_case_methods
class TemporaryFile(File, QROOT.TFile):
    """
    A temporary ROOT file that is automatically deleted when closed.
    Uses Python's :func:`tempfile.mkstemp` to obtain a temporary file
    in the most secure manner possible.

    Positional and keyword arguments are passed directly to
    :func:`tempfile.mkstemp`
    """
    def __init__(self, *args, **kwargs):

        self.__fd, self.__tmp_path = tempfile.mkstemp(*args, **kwargs)
        super(TemporaryFile, self).__init__(self.__tmp_path, 'recreate')

    def Close(self):

        super(TemporaryFile, self).Close()
        os.close(self.__fd)
        os.remove(self.__tmp_path)

    def __exit__(self, type, value, traceback):

        self.Close()
        return False


def root_open(filename, mode=""):

    filename = path.expand(filename)
    root_file = ROOT.TFile.Open(filename, mode)
    # fix evil segfault after attempt to open bad file in 5.30
    # this fix is not needed in 5.32
    # GetListOfClosedObjects() does not appear until 5.30
    if ROOT.gROOT.GetVersionInt() >= 53000:
        GLOBALS['CLOSEDOBJECTS'] = ROOT.gROOT.GetListOfClosedObjects()
    if not root_file:
        raise IOError("Could not open file: '%s'" % filename)
    root_file.__class__ = File
    root_file._path = filename
    root_file._parent = root_file
    return root_file


def open(filename, mode=""):

    warnings.warn("Use root_open instead; open is deprecated.",
                  DeprecationWarning)
    return root_open(filename, mode)
