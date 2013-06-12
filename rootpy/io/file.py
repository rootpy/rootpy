# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module enhances IO-related ROOT functionality
"""
import ROOT

from ..core import Object
from ..decorators import snake_case_methods
from ..context import preserve_current_directory
from .. import asrootpy, QROOT
from . import utils, DoesNotExist
from ..util.path import expand as expand_path

from rootpy import log
from rootpy.memory.keepalive import keepalive

import tempfile
import os
import warnings
import re
from fnmatch import fnmatch

# http://en.wikipedia.org/wiki/Autovivification#Python
from collections import defaultdict
def autovivitree():
    return defaultdict(autovivitree)

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

    def get(self, name, rootpy=True, **kwargs):

        _name = os.path.normpath(name)
        if _name == '.':
            return self
        if _name == '..':
            return self._parent
        try:
            dirpath, _, path = _name.partition(os.path.sep)
            if path:
                if dirpath == '..':
                    return self._parent.Get(path, rootpy=rootpy, **kwargs)
                else:
                    _dir = f(self, dirpath)
                    if not isinstance(_dir, _DirectoryBase):
                        raise DoesNotExist
                    _dir._parent = self
                    _dir._path = os.path.join(self._path, dirpath)
                    thing = _dir.Get(path, rootpy=rootpy, **kwargs)
            else:
                thing = f(self, _name, rootpy=rootpy, **kwargs)
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


def root_open(filename, mode=""):

    filename = expand_path(filename)
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
    root_file._inited = True
    return root_file


def open(filename, mode=""):

    warnings.warn("Use root_open instead; open is deprecated.",
                  DeprecationWarning)
    return root_open(filename, mode)


class _DirectoryBase(Object):

    def walk(self, top=None, class_pattern=None,
             return_classname=False, treat_dirs_as_objs=False):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top, class_pattern=class_pattern,
                          return_classname=return_classname,
                          treat_dirs_as_objs=treat_dirs_as_objs)

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
            raise AttributeError("{0} has no attribute '{1}'".format(self, attr))
        thing = asrootpy(thing)
        if isinstance(thing, Directory):
            thing._path = os.path.join(self._path, thing.GetName())
            thing._parent = self
        return thing

    def __setattr__(self, attr, value):

        if ('_inited' not in self.__dict__ or
            attr in self.__dict__ or
            not isinstance(value, ROOT.TObject)):
            return super(_DirectoryBase, self).__setattr__(attr, value)

        self.__setitem__(attr, value)

    def __getitem__(self, name):

        return self.Get(name)

    def __setitem__(self, name, thing):
        """
        Allow writing objects in a file with ``myfile['thing'] = myobject``
        """
        with preserve_current_directory():
            self.cd()
            thing.Write(name)

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
    def Get(self, name, rootpy=True, **kwargs):
        """
        Return the requested object cast as its corresponding subclass in
        rootpy if one exists and ``rootpy=True``, otherwise return the
        unadulterated TObject.
        """
        thing = super(_DirectoryBase, self).Get(name)
        if not thing:
            raise DoesNotExist
        
        # Ensure that the file we took the object from is alive at least as long
        # as the object being taken from it.
        
        # Note, Python does *not* own `thing`, it is ROOT's responsibility to
        # delete it in the C++ sense. (SetOwnership is False). However, ROOT
        # will delete the object when the TFile's destructor is run.
        # Therefore, when `thing` goes out of scope and the file referred to
        # by `this` has no references left, the file is destructed and calls
        # `thing`'s delete.
        
        # (this is thanks to the fact that weak referents (used by keepalive)
        #  are notified when they are dead).
        
        keepalive(thing, self)
        
        if rootpy:
            return asrootpy(thing, **kwargs)
        return thing

    @wrap_path_handling
    def GetDirectory(self, name, rootpy=True, **kwargs):

        rdir = super(_DirectoryBase, self).GetDirectory(name)
        if not rdir:
            raise DoesNotExist
        if rootpy:
            return asrootpy(rdir, **kwargs)
        return rdir


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
        self._inited = True

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

    # Override .Open
    open = staticmethod(root_open)
    Open = staticmethod(root_open)

    def __init__(self, name, *args, **kwargs):

        # trigger finalSetup
        ROOT.kTRUE
        super(File, self).__init__(name, *args, **kwargs)
        self._post_init()
        
    def _post_init(self):
        self._path = self.GetName()
        self._parent = self
        self._inited = True

    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):

        self.Close()
        return False

    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()

    def _populate_cache(self):

        """
         walk through the whole file and populate the cache
         all objects below the current path are added, i.e.
         for the contents with ina, inb and inab TH1F histograms:

         /a/ina
         /b/inb
         /a/b/inab

         the cache is (omitting the directories):

         cache[""]["obj"] = [("a", ("ina", "TH1F")), ("b", ("inb", "TH1F")), ("a/b", ("inab", "TH1F"))]

         ...

         cache[""]["a"]["b"]["obj"] = [("a/b", ("inab", "TH1F"))]
        """

        self.cache = autovivitree()

        for path, dirs, objects in self.walk(return_classname=True,
                                             treat_dirs_as_objs=True):

            b = self.cache

            for d in [""]+path.split('/'):

                b = b[d]

                obj = [(path, o) for o in objects]

                if "obj" in b:
                    b["obj"] += obj
                else:
                    b["obj"] = obj

    def find(self,
             regexp, negate_regexp=False,
             class_pattern=None,
             find_fnc=re.search,
             refresh_cache=False):
        """

        yield the full path of the matching regular expression and the
        match itself

        """

        if refresh_cache or not hasattr(self,"cache"):
            self._populate_cache()

        b = self.cache

        split_regexp = regexp.split('/')

        # traverse as deep as possible in the cache
        # special case if the first character is not the root, i.e. not ""
        if split_regexp[0] == "":

            for d in split_regexp:

                if d in b:
                    b = b[d]
                else:
                    break

        else:
            b = b[""]

        # perform the search

        for path, (obj, classname) in b["obj"]:

            if class_pattern:
                if not fnmatch(classname, class_pattern):
                    continue

            joined_path = os.path.join(*['/',path,obj])

            result = find_fnc(regexp,joined_path)

            if (result != None) ^ negate_regexp:

                yield joined_path, result

@snake_case_methods
class TemporaryFile(File, QROOT.TFile):
    """
    A temporary ROOT file that is automatically deleted when closed.
    Uses Python's :func:`tempfile.mkstemp` to obtain a temporary file
    in the most secure manner possible.

    Keyword arguments are passed directly to :func:`tempfile.mkstemp`
    """
    def __init__(self, suffix='.root', **kwargs):

        self.__fd, self.__tmp_path = tempfile.mkstemp(suffix=suffix, **kwargs)
        super(TemporaryFile, self).__init__(self.__tmp_path, 'recreate')

    def Close(self):

        super(TemporaryFile, self).Close()
        os.close(self.__fd)
        os.remove(self.__tmp_path)

    def __exit__(self, type, value, traceback):

        self.Close()
        return False
