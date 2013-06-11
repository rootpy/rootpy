# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module contains os.path/walk-like
utilities for the ROOT TFile 'filesystem'
"""
import ROOT
from fnmatch import fnmatch
import itertools
import os
from .. import asrootpy, gDirectory
from ..context import preserve_current_directory
from . import DoesNotExist


def walk(tdirectory, top=None, path=None, depth=0, maxdepth=-1,
         class_pattern=None, return_classname=False, treat_dirs_as_objs=False):
    """
    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple

    dirpath, dirnames, filenames

    dirpath is a string, the path to the directory.  dirnames is a list of the
    names of the subdirectories in dirpath (excluding '.' and '..').  filenames
    is a list of the names of the non-directory files/objects in dirpath.  Note
    that the names in the lists are just names, with no path components.  To get
    a full path (which begins with top) to a file or directory in dirpath, do
    os.path.join(dirpath, name).

    If return_classname is True, each entry in filenames is a tuple of
    the form (filename, classname).

    If treat_dirs_as_objs is True, filenames contains directories as well.

    """

    dirnames, objectnames = [], []
    if top:
        tdirectory = tdirectory.GetDirectory(top)
    for key in tdirectory.unique_keys():
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
        for x in walk(tdirectory.GetDirectory(dirname),
                      class_pattern=class_pattern,
                      depth=depth + 1,
                      maxdepth=maxdepth,
                      path=dirpath,
                      return_classname=return_classname,
                      treat_dirs_as_objs=treat_dirs_as_objs):
            yield x


def splitfile(path):

    filename, _, path = path.partition(':' + os.path.sep)
    return filename, os.path.sep + path


def rm(path, cycle=';*', dir=None):
    ''' Delete an object in a TDirectory

    You must pass the path string relative to dir, otherwise relative to the
    CWD if dir is None.
    '''
    if dir is None:
        dir = gDirectory()
    with preserve_current_directory():
        dirname, objname = os.path.split(os.path.normpath(path))
        if dirname:
            dir = dir.Get(dirname)
        dir.Delete(objname + cycle)


# This is not trivial as I can't figure out a way to delete using the same
# interface as cp (i.e. root objects, not paths)
#def mv(src, dest_dir, newname=None): pass

def cp(src, dest_dir, newname=None, exclude=None, dir=None):
    ''' Copy an object into another TDirectory.

    [src] or [dest_dir] can either be passed as path strings, or as
    ROOT objects themselves.

    If <src> is a TDirectory, the objects will be copied recursively.

    [exclude] can optionally be a function which takes (path, object_name)
    and if returns True excludes objects from being copied.

    The copied object can optionally be given a [newname].

    '''
    if dir is None:
        dir = gDirectory()
    with preserve_current_directory():
        if isinstance(src, basestring):
            src = asrootpy(dir.Get(src))
        if isinstance(dest_dir, basestring):
            dest_dir = asrootpy(dir.Get(dest_dir))
        # Check if the object we are copying is not a directory. Then this is easy
        if not isinstance(src, ROOT.TDirectory):
            if newname is not None:
                src.SetName(newname)
            dest_dir.cd()
            if isinstance(src, ROOT.TTree):
                new_src = src.CloneTree(-1, "fast")
                new_src.Write()
            else:
                src.Write()
        else:
            # We need to copy a directory
            cp_name = src.GetName()
            if newname is not None:
                cp_name = newname
            # See if the directory already exists
            try:
                new_dir = dest_dir.Get(cp_name)
                if not new_dir:
                    raise DoesNotExist
            except DoesNotExist:
                # It doesn't exist, so make the new directory in the destination
                new_dir = dest_dir.mkdir(cp_name)
            # Copy everything in the src directory to the destination directory
            for (path, dirnames, objects) in walk(src, maxdepth=1):
                # Copy all the objects
                for object_name in itertools.chain(objects, dirnames):
                    if exclude and exclude(path, object_name):
                        continue
                    object = asrootpy(src.Get(object_name))
                    # Recursively copy the sub-objects into the dest. dir
                    cp(object, new_dir, dir=dir)


def mkdir(dest_path, recurse=False, dir=None):
    ''' Make a new directory

    If recurse is True, create parent directories as required.

    Return the newly created TDirectory
    '''
    if dir is None:
        dir = gDirectory()
    with preserve_current_directory():
        head, tail = os.path.split(os.path.normpath(dest_path))
        dest = dir
        if tail == "":
            raise ValueError("invalid directory name: %s" % dest_path)
        if recurse:
            parent_dirs = head.split('/')
            for parent_dir in parent_dirs:
                try:
                    newdest = dest.GetDirectory(parent_dir)
                    if not newdest:
                        raise DoesNotExist
                    newdest.cd()
                    dest = newdest
                except DoesNotExist:
                    dest = dest.mkdir(parent_dir)
                    dest.cd()
        elif head != "":
            dest = dest.GetDirectory(head)
            if not dest:
                raise DoesNotExist(head)
            dest.cd()
        try:
            if dest.GetDirectory(tail):
                raise ValueError("%s already exists" % dest_path)
        except DoesNotExist:
            pass
        dest = dest.mkdir(tail)
    return dest
