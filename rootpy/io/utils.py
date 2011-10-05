"""
This module contains os.path/walk-like utilities for the ROOT TFile 'filesystem'
"""


def walk(tdirectory, top=None):
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
    keys = tdirectory.GetListOfKeys()
    for key in keys:
        name = key.GetName()
        classname = key.GetClassName()
        # print name, classname
        if 'TDirectory' in classname:
            dirnames.append(name)
        else:
            objectnames.append(name)
    yield tdirectory, dirnames, objectnames
    for dirname in dirnames:
        for x in walk(tdirectory.GetDirectory(dirname)):
            yield x
