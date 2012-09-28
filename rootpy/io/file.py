"""
This module enhances IO-related ROOT functionality
"""
import ROOT
from ..core import snake_case_methods
from ..registry import register
from ..utils import asrootpy
from . import utils
from .. import path
from .. import rootpy_globals
import tempfile
import os


__all__ = [
    'DoesNotExist',
    'Directory',
    'File',
    'TemporaryFile',
    'open',
]


VALIDPATH = '^(?P<file>.+.root)(?:[/](?P<path>.+))?$'
GLOBALS = {}


class DoesNotExist(Exception):
    pass


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


class _DirectoryBase(object):
    """
    A mixin (can't stand alone). To be improved.
    """

    def walk(self, top=None, class_pattern=None):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top, class_pattern=class_pattern)

    def __getattr__(self, attr):
        """
        Natural naming support.
        Now you can get an object from a File/Directory with
        myfile.somedir.otherdir.histname

        Must be careful here... if __getattr__ ends up being called
        in Get this can end up in an "infinite" recursion and stack overflow
        """
        return self.Get(attr)

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
        thing = asrootpy(self.__class__.__bases__[-1].Get(self, name),
                         **kwargs)
        if not thing:
            raise DoesNotExist
        return thing

    @wrap_path_handling
    def GetDirectory(self, name, **kwargs):
        """
        Return a Directory object rather than TDirectory
        """
        dir = asrootpy(self.__class__.__bases__[-1].GetDirectory(self, name),
                       **kwargs)
        if not dir:
            raise DoesNotExist
        return dir

    def cd(self, *args):

        rootpy_globals.directory = self
        self.__class__.__bases__[-1].cd(self, *args)


@snake_case_methods
@register()
class Directory(_DirectoryBase, ROOT.TDirectoryFile):
    """
    Inherits from TDirectory
    """

    def __init__(self, name, title, *args, **kwargs):

        ROOT.TDirectoryFile.__init__(self, name, title, *args)
        self._path = name
        self._parent = None
        rootpy_globals.directory = self

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()


@snake_case_methods
class File(_DirectoryBase, ROOT.TFile):
    """
    Wrapper for TFile that adds various convenience functions.

    >>> from rootpy.test import filename
    >>> f = File(filename, 'read')

    """

    def __init__(self, *args, **kwargs):

        ROOT.TFile.__init__(self, *args, **kwargs)
        self._path = self.GetName()
        self._parent = self
        rootpy_globals.directory = self

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
class TemporaryFile(File, ROOT.TFile):

    def __init__(self, *args, **kwargs):

        self.__fd, path = tempfile.mkstemp(*args, **kwargs)
        super(TemporaryFile, self).__init__(path, 'recreate')

    def Close(self):

        super(TemporaryFile, self).Close()
        os.close(self.__fd)

    def __exit__(self, type, value, traceback):

        self.Close()
        os.unlink(self.GetName())
        return False


def open(filename, mode=""):

    filename = path.expand(filename)
    file = ROOT.TFile.Open(filename, mode)
    # fix evil segfault after attempt to open bad file in 5.30
    # this fix is not needed in 5.32
    # GetListOfClosedObjects() does not appear until 5.30
    if ROOT.gROOT.GetVersionInt() >= 53000:
        GLOBALS['CLOSEDOBJECTS'] = ROOT.gROOT.GetListOfClosedObjects()
    if not file:
        raise IOError("Could not open file: '%s'" % filename)
    file.__class__ = File
    file._path = filename
    file._parent = file
    rootpy_globals.directory = file
    return file
