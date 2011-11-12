"""
This module contains os.path/walk-like
utilities for the ROOT TFile 'filesystem'
"""
from fnmatch import fnmatch
import os
import ROOT

def walk(tdirectory, top=None, path=None, depth=0, maxdepth=-1, class_pattern=None):
    """
    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple

    dirpath, dirnames, filenames

    dirpath is a string, the path to the directory.  dirnames is a list of
    the names of the subdirectories in dirpath (excluding '.' and '..').
    filenames is a list of the names of the non-directory files in dirpath.
    Note that the names in the lists are just names, with no path components.
    To get a full path (which begins with top) to a file or directory in
    dirpath, do os.path.join(dirpath, name).
    """
    dirnames, objectnames = [], []
    if top:
        tdirectory = tdirectory.GetDirectory(top)
    for key in tdirectory.GetListOfKeys():
        name = key.GetName()
        classname = key.GetClassName()
        # print name, classname
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
                      depth=depth+1,
                      maxdepth=maxdepth,
                      path=dirpath):
            yield x


def splitfile(path):

    filename, _, path = path.partition(':' + os.path.sep)
    return filename, os.path.sep + path
