# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import glob
import os
import errno

__all__ = [
    'expand',
    'expand_and_glob',
    'expand_and_glob_all',
    'mkdir_p',
]


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def expand_and_glob(s):
    return glob.glob(expand(s))


def expand_and_glob_all(s):
    files = []
    for name in s:
        files += expand_and_glob(name)
    return files


def mkdir_p(path):
    """
    mkdir -p functionality
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    In rootpy, this function should be used when creating directories in a
    multithreaded environment to avoid race conditions when checking if a
    directory exists before creating it.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
