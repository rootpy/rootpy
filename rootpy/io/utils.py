"""
This module contains os.path/walk-like
utilities for the ROOT TFile 'filesystem'
"""
import ROOT
from fnmatch import fnmatch
import itertools
import os
from ..utils import asrootpy
from . import DoesNotExist


def walk(tdirectory, top=None, path=None, depth=0, maxdepth=-1, class_pattern=None):
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
    """
    dirnames, objectnames = [], []
    if top:
        tdirectory = tdirectory.GetDirectory(top)
    for key in tdirectory.unique_keys():
        name = key.GetName()
        classname = key.GetClassName()
        if classname.startswith('TDirectory'):
            dirnames.append(name)
        else:
            if class_pattern is not None:
                if not fnmatch(classname, class_pattern):
                    continue
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
                      path=dirpath):
            yield x


def splitfile(path):

    filename, _, path = path.partition(':' + os.path.sep)
    return filename, os.path.sep + path


def rm(path_to_object, cycle=';*'):
    ''' Delete an object in a TDirectory

    You must pass the path string, relative to the CWD
    '''
    # Save state
    current_path = ROOT.gDirectory.GetPathStatic()
    # Get the location of the thing to be deleted
    dirname = os.path.dirname(path_to_object)
    objname = os.path.basename(path_to_object)
    ROOT.gDirectory.cd(dirname)
    ROOT.gDirectory.Delete(objname + cycle)
    # Restore state
    ROOT.gDirectory.cd(current_path)


# This is not trivial as I can't figure out a way to delete using the same
# interface as cp (i.e. root objects, not paths)
#def mv(src, dest_dir, newname=None): pass

def cp(src, dest_dir, newname=None, exclude=None):
    ''' Copy an object into another TDirectory.

    [src] or [dest_dir] can either be passed as path strings, or as
    ROOT objects themselves.

    If <src> is a TDirectory, the objects will be copied recursively.

    [exclude] can optionally be a function which takes (path, object_name)
    and if returns True excludes objects from being copied.

    The copied object can optionally be given a [newname].

    '''
    # Always save/restore the state of gDirectory.  Have to use the string path
    # otherwise gDirectory will change out from under us
    current_path = ROOT.gDirectory.GetPathStatic()

    if isinstance(src, basestring):
        src = asrootpy(ROOT.gDirectory.Get(src))
    if isinstance(dest_dir, basestring):
        dest_dir = asrootpy(ROOT.gDirectory.Get(dest_dir))

    # Check if the object we are copying is not a directory.  Then this is easy
    if not isinstance(src, ROOT.TDirectory):
        if newname is not None:
            src.SetName(newname)
        dest_dir.cd()
        src.Write()
    else:  # We need to copy a directory
        cp_name = src.GetName()
        if newname is not None:
            cp_name = newname
        # See if the directory already exists
        new_dir = dest_dir.Get(cp_name)
        if not new_dir:
            # It doesn't exist, make the new directory in the destination
            new_dir = dest_dir.mkdir(cp_name)
        # Copy everything in the src directory to the destination directory
        for (path, dirnames, objects) in walk(src, maxdepth=1):
            # Copy all the objects
            for object_name in itertools.chain(objects, dirnames):
                if exclude and exclude(path, object_name):
                    continue
                object = asrootpy(src.Get(object_name))
                # Recursively copy the sub-objects into the dest. dir
                cp(object, new_dir)
    # Restore the state when done
    ROOT.gDirectory.cd(current_path)


def mkdir(dest_path, recurse=False):
    ''' Make a new directory relative to the CWD

    If recurse is True, create parent directories as required.

    Return the newly created TDirectory
    '''
    # Always save/restore the state of gDirectory.  Have to use the string path
    # otherwise gDirectory will change out from under us
    current_path = ROOT.gDirectory.GetPathStatic()
    head, tail = os.path.split(os.path.normpath(dest_path))
    if tail == "":
        raise ValueError("invalid directory name: %s" % dest_path)
    if recurse:
        parent_dirs = head.split('/')
        for parent_dir in parent_dirs:
            try:
                dest = ROOT.gDirectory.GetDirectory(parent_dir)
                if not dest:
                    raise DoesNotExist
                dest.cd()
            except DoesNotExist:
                dest = ROOT.gDirectory.mkdir(parent_dir)
                dest.cd()
    elif head != "":
        dest = ROOT.gDirectory.GetDirectory(head)
        if not dest:
            raise DoesNotExist(head)
        dest.cd()
    try:
        dest = ROOT.gDirectory.GetDirectory(tail)
        if dest:
            raise ValueError("%s already exists" % dest_path)
    except DoesNotExist:
        pass
    dest = ROOT.gDirectory.mkdir(tail)
    # Restore the state when done
    ROOT.gDirectory.cd(current_path)
    return dest
