# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import glob
import os


def expand(s):

    return os.path.expanduser(os.path.expandvars(s))


def expand_and_glob(s):

    return glob.glob(expand(s))


def expand_and_glob_all(s):

    files = []
    for name in s:
        files += expand_and_glob(name)
    return files
