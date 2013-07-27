# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module enhances IO-related ROOT functionality
"""
import ROOT

from ..core import Object, NamedObject
from ..decorators import snake_case_methods
from ..context import preserve_current_directory
from .. import asrootpy, QROOT, gDirectory
from ..util.path import expand as expand_path

from rootpy import log
from rootpy.memory.keepalive import keepalive

import tempfile
import os
import warnings
import itertools
import re
from fnmatch import fnmatch
from collections import defaultdict


__all__ = [
    'DoesNotExist',
    'Key',
    'Directory',
    'File',
    'MemFile',
    'TemporaryFile',
    'root_open',
]


VALIDPATH = '^(?P<file>.+.root)(?:[/](?P<path>.+))?$'
GLOBALS = {}


class DoesNotExist(Exception):
    pass


def autovivitree():
    # http://en.wikipedia.org/wiki/Autovivification#Python
    return defaultdict(autovivitree)


def splitfile(path):

    filename, _, path = path.partition(':' + os.path.sep)
    return filename, os.path.sep + path


def wrap_path_handling(f):

    def get(self, name, *args, **kwargs):

        _name = os.path.normpath(name)
        if _name == '.':
            return self
        if _name == '..':
            return self._parent
        try:
            dirpath, _, path = _name.partition(os.path.sep)
            if path:
                if dirpath == '..':
                    return self._parent.Get(path, *args, **kwargs)
                else:
                    _dir = self.Get(dirpath)
                    if not isinstance(_dir, _DirectoryBase):
                        raise DoesNotExist
                    _dir._parent = self
                    _dir._path = os.path.join(self._path, dirpath)
                    thing = f(_dir, path, *args, **kwargs)
            else:
                thing = f(self, _name, *args, **kwargs)
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
            raise DoesNotExist(
                "requested path '{0}' does not exist in {1}".format(
                    name, self._path))
    return get


def root_open(filename, mode=''):

    filename = expand_path(filename)
    root_file = ROOT.TFile.Open(filename, mode)
    # fix evil segfault after attempt to open bad file in 5.30
    # this fix is not needed in 5.32
    # GetListOfClosedObjects() does not appear until 5.30
    if ROOT.gROOT.GetVersionInt() >= 53000:
        GLOBALS['CLOSEDOBJECTS'] = ROOT.gROOT.GetListOfClosedObjects()
    if not root_file:
        raise IOError("could not open file: '{0}'".format(filename))
    root_file.__class__ = File
    root_file._path = filename
    root_file._parent = root_file
    root_file._inited = True
    return root_file


@snake_case_methods
class Key(NamedObject, QROOT.TKey):
    pass


class _DirectoryBase(Object):

    def __str__(self):

        return "{0}('{1}')".format(self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()

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
            raise AttributeError(
                "{0} has no attribute '{1}'".format(self, attr))
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

    def objects(self, cls=None):
        """
        Return an iterater over all objects in this directory which are
        instances of `cls`. By default, iterate over all objects (`cls=None`).

        Example usage:

            $ rootpy browse myfile.root

            In [1]: list(f1.objects(R.Directory))
            Out[1]: [Directory('mydirectory')]

        """
        objs = (asrootpy(x.ReadObj(), warn=False)
                for x in self.GetListOfKeys())
        if cls is not None:
            objs = (obj for obj in objs if isinstance(obj, cls))
        return objs

    def keys(self):
        """
        Return a list of the keys in this directory.
        """
        return [asrootpy(key) for key in self.GetListOfKeys()]

    def latest_keys(self):
        """
        Return a list of keys with unique names where only the key with the
        highest cycle number is included where multiple keys exist with the
        same name.
        """
        keys = {}
        for key in self.keys():
            name = key.GetName()
            if name in keys:
                if key.GetCycle() > keys[name].GetCycle():
                    keys[name] = key
            else:
                keys[name] = key
        return keys.values()

    @wrap_path_handling
    def Get(self, path, rootpy=True, **kwargs):
        """
        Return the requested object cast as its corresponding subclass in
        rootpy if one exists and ``rootpy=True``, otherwise return the
        unadulterated TObject.
        """
        thing = super(_DirectoryBase, self).Get(path)
        if not thing:
            raise DoesNotExist

        # Ensure that the file we took the object from is alive at least as
        # long as the object being taken from it.

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
    def GetDirectory(self, path, rootpy=True, **kwargs):

        rdir = super(_DirectoryBase, self).GetDirectory(path)
        if not rdir:
            raise DoesNotExist
        if rootpy:
            return asrootpy(rdir, **kwargs)
        return rdir

    @wrap_path_handling
    def GetKey(self, path, cycle=9999, rootpy=True, **kwargs):
        """
        Override TDirectory's GetKey and also handle accessing keys nested
        arbitrarily deep in subdirectories.
        """
        key = super(_DirectoryBase, self).GetKey(path, cycle)
        if not key:
            raise DoesNotExist
        if rootpy:
            return asrootpy(key, **kwargs)
        return key

    def __contains__(self, path):
        """
        Determine if a an object exists in the file at the path `path`::

            if 'some/thing' in file:
                # do something
        """
        try:
            self.GetKey(path)
            return True
        except DoesNotExist:
            return False

    def mkdir(self, path, title="", recurse=False):
        """
        Make a new directory. If recurse is True, create parent directories
        as required. Return the newly created TDirectory.
        """
        head, tail = os.path.split(os.path.normpath(path))
        if tail == "":
            raise ValueError("invalid directory name: {0}".format(path))
        with preserve_current_directory():
            dest = self
            if recurse:
                parent_dirs = head.split('/')
                for parent_dir in parent_dirs:
                    try:
                        newdest = dest.GetDirectory(parent_dir)
                        dest = newdest
                    except DoesNotExist:
                        dest = dest.mkdir(parent_dir)
            elif head != "":
                dest = dest.GetDirectory(head)
            if tail in dest:
                raise ValueError("{0} already exists".format(path))
            newdir = asrootpy(super(_DirectoryBase, dest).mkdir(tail, title))
        return newdir

    def rm(self, path, cycle=';*'):
        """
        Delete an object at `path` relative to this directory
        """
        rdir = self
        with preserve_current_directory():
            dirname, objname = os.path.split(os.path.normpath(path))
            if dirname:
                rdir = rdir.Get(dirname)
            rdir.Delete(objname + cycle)

    # TODO:
    # def move(self, src, dest, newname=None):

    def copytree(self, dest_dir, src=None, newname=None,
                 exclude=None, overwrite=False):
        """
        Copy this directory or just one contained object into another
        directory.

        `dest_dir` can either be the string path or a Directory.

        If `src` is None then this entire directory is copied recursively
        otherwise if `src` is a string path to an object relative to this
        directory, only that object will be copied. The copied object can
        optionally be given a `newname`.

        `exclude` can optionally be a function which takes (path, object_name)
        and if returns True excludes objects from being copied if the entire
        directory is being copied recursively.
        """
        def copy_object(obj, dest, name=None):
            if name is None:
                name = obj.GetName()
            if not overwrite and name in dest:
                raise ValueError(
                    "{0} already exists in {1} and `overwrite=False`".format(
                        name, dest._path))
            dest.cd()
            if isinstance(obj, ROOT.TTree):
                new_obj = obj.CloneTree(-1, "fast")
                new_obj.Write(name, ROOT.TObject.kOverwrite)
            else:
                obj.Write(name, ROOT.TObject.kOverwrite)

        with preserve_current_directory():
            if isinstance(src, basestring):
                src = asrootpy(self.Get(src))
            else:
                src = self
            if isinstance(dest_dir, basestring):
                try:
                    dest_dir = asrootpy(self.GetDirectory(dest_dir))
                except DoesNotExist:
                    dest_dir = self.mkdir(dest_dir)
            if isinstance(src, ROOT.TDirectory):
                # Copy a directory
                cp_name = newname if newname is not None else src.GetName()
                # See if the directory already exists
                if cp_name not in dest_dir:
                    # Destination directory doesn't exist, so make a new one
                    new_dir = dest_dir.mkdir(cp_name)
                # Copy everything in the src directory to the destination
                for (path, dirnames, objects) in src.walk(maxdepth=0):
                    # Copy all the objects
                    for object_name in objects:
                        if exclude and exclude(path, object_name):
                            continue
                        thing = src.Get(object_name)
                        copy_object(thing, new_dir)
                    for dirname in dirnames:
                        if exclude and exclude(path, dirname):
                            continue
                        rdir = src.GetDirectory(dirname)
                        # Recursively copy objects in subdirectories
                        rdir.copytree(
                            new_dir,
                            exclude=exclude, overwrite=overwrite)
            else:
                # Copy an object
                copy_object(src, dest_dir, name=newname)

    def walk(self, top=None, path=None, depth=0, maxdepth=-1,
             class_pattern=None, return_classname=False,
             treat_dirs_as_objs=False):
        """
        For each directory in the directory tree rooted at top (including top
        itself, but excluding '.' and '..'), yields a 3-tuple::

            dirpath, dirnames, filenames

        `dirpath` is a string, the path to the directory.  dirnames is a list
        of the names of the subdirectories in `dirpath`
        (excluding '.' and '..').

        `filenames` is a list of the names of the non-directory files/objects
        in `dirpath`.

        Note that the names in the lists are just names, with no
        path components.  To get a full path (which begins with top) to a file
        or directory in `dirpath`, do `os.path.join(dirpath, name)`.

        If `return_classname` is True, each entry in `filenames` is a tuple of
        the form `(filename, classname)`.

        If `treat_dirs_as_objs` is True, `filenames` contains directories
        as well.

        """
        dirnames, objectnames = [], []
        tdirectory = self.GetDirectory(top) if top else self
        for key in tdirectory.latest_keys():
            name = key.GetName()
            classname = key.GetClassName()
            is_directory = classname.startswith('TDirectory')
            if is_directory:
                dirnames.append(name)
            if not is_directory or treat_dirs_as_objs:
                if class_pattern is not None:
                    if not fnmatch(classname, class_pattern):
                        continue
                name = (name if not return_classname else (name, classname))
                objectnames.append(name)
        if path:
            dirpath = os.path.join(path, tdirectory.GetName())
        elif not isinstance(tdirectory, ROOT.TFile):
            dirpath = tdirectory.GetName()
        else:
            dirpath = ''
        yield dirpath, dirnames, objectnames
        if depth == maxdepth:
            return
        for dirname in dirnames:
            rdir = tdirectory.GetDirectory(dirname)
            for x in rdir.walk(
                    class_pattern=class_pattern,
                    depth=depth + 1,
                    maxdepth=maxdepth,
                    path=dirpath,
                    return_classname=return_classname,
                    treat_dirs_as_objs=treat_dirs_as_objs):
                yield x


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


@snake_case_methods
class _FileBase(_DirectoryBase):

    def __init__(self, name, *args, **kwargs):

        # trigger finalSetup
        ROOT.kTRUE
        super(_FileBase, self).__init__(name, *args, **kwargs)
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

    def _populate_cache(self):

        """
        Walk through the whole file and populate the cache
        all objects below the current path are added, i.e.
        for the contents with ina, inb and inab TH1F histograms::

           /a/ina
           /b/inb
           /a/b/inab

        the cache is (omitting the directories)::

            cache[""]["obj"] = [("a", ("ina", "TH1F")),
                                ("b", ("inb", "TH1F")),
                                ("a/b", ("inab", "TH1F"))]

            ...

            cache[""]["a"]["b"]["obj"] = [("a/b", ("inab", "TH1F"))]

        """

        self.cache = autovivitree()

        for path, dirs, objects in self.walk(return_classname=True,
                                             treat_dirs_as_objs=True):
            b = self.cache
            for d in ['']+path.split('/'):
                b = b[d]
                obj = [(path, o) for o in objects]
                if 'obj' in b:
                    b['obj'] += obj
                else:
                    b['obj'] = obj

    def find(self,
             regexp, negate_regexp=False,
             class_pattern=None,
             find_fnc=re.search,
             refresh_cache=False):
        """
        yield the full path of the matching regular expression and the
        match itself
        """
        if refresh_cache or not hasattr(self, 'cache'):
            self._populate_cache()

        b = self.cache
        split_regexp = regexp.split('/')

        # traverse as deep as possible in the cache
        # special case if the first character is not the root, i.e. not ""
        if split_regexp[0] == '':
            for d in split_regexp:
                if d in b:
                    b = b[d]
                else:
                    break
        else:
            b = b['']

        # perform the search
        for path, (obj, classname) in b['obj']:
            if class_pattern:
                if not fnmatch(classname, class_pattern):
                    continue
            joined_path = os.path.join(*['/', path, obj])
            result = find_fnc(regexp, joined_path)
            if (result is not None) ^ negate_regexp:
                yield joined_path, result


@snake_case_methods
class File(_FileBase, QROOT.TFile):
    """
    A subclass of ROOT's TFile adding all of the rootpy goodness.

    >>> from rootpy.test import filename
    >>> f = File(filename, 'read')

    """
    # Override .Open
    open = staticmethod(root_open)
    Open = staticmethod(root_open)


@snake_case_methods
class MemFile(_FileBase, QROOT.TMemFile):
    """
    A subclass of ROOT's TMemFile adding all of the rootpy goodness.

    >>> f = MemFile('test', 'recreate')

    """
    pass


@snake_case_methods
class TemporaryFile(File):
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
        """
        The physical file is automatically deleted after being closed.
        """
        super(TemporaryFile, self).Close()
        os.close(self.__fd)
        os.remove(self.__tmp_path)
