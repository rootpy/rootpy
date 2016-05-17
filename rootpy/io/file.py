# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module enhances IO-related ROOT functionality
"""
from __future__ import absolute_import

import os
import re
import tempfile
from fnmatch import fnmatch
from collections import defaultdict

from .. import ROOT
from .. import asrootpy, QROOT
from ..base import Object, NamedObject
from ..decorators import snake_case_methods
from ..context import preserve_current_directory
from ..utils.path import expand as expand_path
from ..memory.keepalive import keepalive
from ..extern.shortuuid import uuid
from ..extern.six import string_types


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


class DoesNotExist(Exception):
    """
    This exception is raised if an attempt is made to access an object
    that does not exist in a directory.
    """
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
                    thing = get(_dir, path, *args, **kwargs)
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
    """
    Open a ROOT file via ROOT's static ROOT.TFile.Open [1] function and return
    an asrootpy'd File.

    Parameters
    ----------

    filename : string
        The absolute or relative path to the ROOT file.

    mode : string, optional (default='')
        Mode indicating how the file is to be opened.  This can be either one
        of the options supported by ROOT.TFile.Open [2], or one of `a`, `a+`,
        `r`, `r+`, `w` or `w+`, with meanings as for the built-in `open()`
        function [3].

    Returns
    -------

    root_file : File
        an instance of rootpy's File subclass of ROOT's TFile.

    References
    ----------

    .. [1] http://root.cern.ch/root/html/TFile.html#TFile:Open
    .. [2] http://root.cern.ch/root/html/TFile.html#TFile:TFile@2
    .. [3] https://docs.python.org/2/library/functions.html#open

    """
    mode_map = {'a': 'UPDATE',
                'a+': 'UPDATE',
                'r': 'READ',
                'r+': 'UPDATE',
                'w': 'RECREATE',
                'w+': 'RECREATE'}

    if mode in mode_map:
        mode = mode_map[mode]

    filename = expand_path(filename)
    prev_dir = ROOT.gDirectory.func()
    root_file = ROOT.R.TFile.Open(filename, mode)
    if not root_file:
        raise IOError("could not open file: '{0}'".format(filename))
    root_file.__class__ = File
    root_file._path = filename
    root_file._parent = root_file
    root_file._prev_dir = prev_dir
    root_file._inited = True
    # give Python ownership of the TFile so we can delete it
    ROOT.SetOwnership(root_file, True)
    return root_file


@snake_case_methods
class Key(NamedObject, QROOT.TKey):
    """
    A subclass of ROOT's TKey [1]

    References
    ----------

    .. [1] http://root.cern.ch/root/html/TKey.html

    """
    _ROOT = QROOT.TKey


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
                not isinstance(value, ROOT.R.TObject)):
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
        return self.objects()

    def __enter__(self):
        curr_dir = ROOT.gDirectory.func()
        if curr_dir != self:
            self._prev_dir = curr_dir
        self.cd()
        return self

    def __exit__(self, type, value, traceback):
        self.Close()
        return False

    def cd_previous(self):
        """
        cd to the gDirectory before this file was open.
        """
        if isinstance(self._prev_dir, ROOT.TROOT):
            return False
        if isinstance(self._prev_dir, ROOT.TFile):
            if self._prev_dir.IsOpen() and self._prev_dir.IsWritable():
                self._prev_dir.cd()
                return True
            return False
        if not self._prev_dir.IsWritable():
            # avoid warning from ROOT stating file is not writable
            return False
        prev_file = self._prev_dir.GetFile()
        if prev_file and prev_file.IsOpen():
            self._prev_dir.cd()
            return True
        return False

    def Close(self, *args):
        """
        Like ROOT's Close but reverts to the gDirectory before this file was
        opened.
        """
        super(_DirectoryBase, self).Close(*args)
        return self.cd_previous()

    def objects(self, cls=None):
        """
        Return an iterater over all objects in this directory which are
        instances of `cls`. By default, iterate over all objects (`cls=None`).

        Parameters
        ----------

        cls : a class, optional (default=None)
            If a class is specified, only iterate over objects that are
            instances of this class.

        Returns
        -------

        A generator over the objects in this directory.

        Examples
        --------

            $ rootpy browse myfile.root

            In [1]: list(f1.objects(R.Directory))
            Out[1]: [Directory('mydirectory')]

        """
        objs = (asrootpy(x.ReadObj(), warn=False)
                for x in self.GetListOfKeys())
        if cls is not None:
            objs = (obj for obj in objs if isinstance(obj, cls))
        return objs

    def keys(self, latest=False):
        """
        Return a list of the keys in this directory.

        Parameters
        ----------

        latest : bool, optional (default=False)
            If True then return a list of keys with unique names where only the
            key with the highest cycle number is included where multiple keys
            exist with the same name.

        Returns
        -------

        keys : list
            List of keys

        """
        if latest:
            keys = {}
            for key in self.keys():
                name = key.GetName()
                if name in keys:
                    if key.GetCycle() > keys[name].GetCycle():
                        keys[name] = key
                else:
                    keys[name] = key
            return keys.values()
        return [asrootpy(key) for key in self.GetListOfKeys()]

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
                parent_dirs = head.split(os.path.sep)
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

        Parameters
        ----------

        dest_dir : string or Directory
            The destination directory.

        src : string, optional (default=None)
            If ``src`` is None then this entire directory is copied recursively
            otherwise if ``src`` is a string path to an object relative to this
            directory, only that object will be copied. The copied object can
            optionally be given a ``newname``.

        newname : string, optional (default=None)
            An optional new name for the copied object.

        exclude : callable, optional (default=None)
            ``exclude`` can optionally be a function which takes
            ``(path, object_name)`` and if returns True excludes
            objects from being copied if the entire directory is being copied
            recursively.

        overwrite : bool, optional (default=False)
            If True, then overwrite existing objects with the same name.

        """
        def copy_object(obj, dest, name=None):
            if name is None:
                name = obj.GetName()
            if not overwrite and name in dest:
                raise ValueError(
                    "{0} already exists in {1} and `overwrite=False`".format(
                        name, dest._path))
            dest.cd()
            if isinstance(obj, ROOT.R.TTree):
                new_obj = obj.CloneTree(-1, "fast")
                new_obj.Write(name, ROOT.R.TObject.kOverwrite)
            else:
                obj.Write(name, ROOT.R.TObject.kOverwrite)

        with preserve_current_directory():
            if isinstance(src, string_types):
                src = asrootpy(self.Get(src))
            else:
                src = self
            if isinstance(dest_dir, string_types):
                try:
                    dest_dir = asrootpy(self.GetDirectory(dest_dir))
                except DoesNotExist:
                    dest_dir = self.mkdir(dest_dir)
            if isinstance(src, ROOT.R.TDirectory):
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

    def walk(self,
             top=None,
             path=None,
             depth=0,
             maxdepth=-1,
             class_ref=None,
             class_pattern=None,
             return_classname=False,
             treat_dirs_as_objs=False):
        """
        Walk the directory structure and content in and below a directory.
        For each directory in the directory tree rooted at ``top`` (including
        ``top`` itself, but excluding '.' and '..'), yield a 3-tuple
        ``dirpath, dirnames, filenames``.

        Parameters
        ----------

        top : string, optional (default=None)
            A path to a starting directory relative to this directory,
            otherwise start at this directory.

        path : string, optional (default=None)
            A path prepended as a prefix on the ``dirpath``. This argument is
            used internally as the recursion traverses down through
            subdirectories.

        depth : int, optional (default=0)
            The current depth, used internally as the recursion traverses down
            through subdirectories.

        max_depth : int, optional (default=-1)
            The maximum depth in the directory hierarchy to traverse. There is
            no limit applied by default.

        class_ref : class, optional (default=None)
            If not None then only include objects that are instances of
            ``class_ref``.

        class_pattern : string, optional (default=None)
            If not None then only include objects in ``filenames`` with class
            names that match ``class_pattern``. ``class_pattern`` should be a
            Unix shell-style wildcarded string.

        return_classname : bool, optional (default=False)
            If True, then each entry in ``filenames`` is a tuple of
            the form ``(filename, classname)``.

        treat_dirs_as_objs : bool, optional (default=False)
            If True, ``filenames`` contains directories as well.

        Returns
        -------

        dirpath, dirnames, filenames : iterator
            An iterator over the 3-tuples ``dirpath, dirnames, filenames``.
            ``dirpath`` is a string, the path to the directory. ``dirnames`` is
            a list of the names of the subdirectories in ``dirpath``
            (excluding '.' and '..'). ``filenames`` is a list of the names of
            the non-directory files/objects in ``dirpath``.

        Notes
        -----

        The names in the lists are just names, with no path components.
        To get a full path (which begins with top) to a file or directory
        in ``dirpath``, use ``os.path.join(dirpath, name)``.

        """
        dirnames, objectnames = [], []
        tdirectory = self.GetDirectory(top) if top else self
        for key in tdirectory.keys(latest=True):
            name = key.GetName()
            classname = key.GetClassName()
            is_directory = classname.startswith('TDirectory')
            if is_directory:
                dirnames.append(name)
            if not is_directory or treat_dirs_as_objs:
                if class_ref is not None:
                    tclass = ROOT.TClass.GetClass(classname, True, True)
                    if not tclass or not tclass.InheritsFrom(class_ref.Class()):
                        continue
                if class_pattern is not None:
                    if not fnmatch(classname, class_pattern):
                        continue
                name = (name if not return_classname else (name, classname))
                objectnames.append(name)
        if path:
            dirpath = os.path.join(path, tdirectory.GetName())
        elif not isinstance(tdirectory, ROOT.R.TFile):
            dirpath = tdirectory.GetName()
        else:
            dirpath = ''
        yield dirpath, dirnames, objectnames
        if depth == maxdepth:
            return
        for dirname in dirnames:
            rdir = tdirectory.GetDirectory(dirname)
            for x in rdir.walk(
                    class_ref=class_ref,
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
    A subclass of ROOT's TDirectoryFile [1]

    References
    ----------

    .. [1] http://root.cern.ch/root/html/TDirectoryFile.html

    """
    _ROOT = QROOT.TDirectoryFile

    def __init__(self, name, title=None, classname='', parent=None):
        if title is None:
            title = name
        super(Directory, self).__init__(name, title, classname, parent or 0)
        self._post_init()

    def _post_init(self):
        self._path = self.GetName()
        self._parent = ROOT.gDirectory.func()
        self._prev_dir = None
        self._inited = True


class _FileBase(_DirectoryBase):

    def __init__(self, name, *args, **kwargs):
        # trigger finalSetup
        ROOT.R.kTRUE
        self._prev_dir = ROOT.gDirectory.func()
        super(_FileBase, self).__init__(name, *args, **kwargs)
        self._post_init()

    def _post_init(self):
        self._path = self.GetName()
        self._parent = self
        self._inited = True

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
    A subclass of ROOT's TFile [1]

    Examples
    --------

    >>> from rootpy.io import File
    >>> from rootpy.testdata import get_filepath
    >>> f = File(get_filepath(), 'read')
    >>> list(f)
    [Directory('means'), Directory('scales'), Directory('gaps'), Directory('efficiencies'), Directory('dimensions'), Directory('graphs')]
    >>> f.means
    Directory('rootpy/testdata/test_file.root/means')

    References
    ----------

    .. [1] http://root.cern.ch/root/html/TFile.html

    """
    _ROOT = QROOT.TFile
    # Override .Open
    open = staticmethod(root_open)
    Open = staticmethod(root_open)


@snake_case_methods
class MemFile(_FileBase, QROOT.TMemFile):
    """
    A subclass of ROOT's TMemFile [1]

    Examples
    --------

    >>> from rootpy.io import MemFile
    >>> f = MemFile()

    References
    ----------

    .. [1] http://root.cern.ch/root/html/TMemFile.html

    """
    _ROOT = QROOT.TMemFile

    def __init__(self, name=None, mode='recreate'):
        if name is None:
            name = '{0}_{1}'.format(self.__class__.__name__, uuid())
        super(MemFile, self).__init__(name, mode)


@snake_case_methods
class TemporaryFile(File):
    """
    A temporary ROOT file that is automatically deleted when closed.
    Python's :func:`tempfile.mkstemp` [1] is used to obtain a temporary file
    in the most secure manner possible.

    Keyword arguments are passed directly to :func:`tempfile.mkstemp` [1]

    References
    ----------

    .. [1] http://docs.python.org/2/library/tempfile.html#tempfile.mkstemp

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
