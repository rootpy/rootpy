# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module handles creation of the user-data area
"""
from __future__ import absolute_import

import os
import tempfile
import atexit
from os.path import expanduser, expandvars, exists, isdir, join as pjoin
from platform import machine

from . import log; log = log[__name__]
from . import QROOT, IN_NOSETESTS
from .defaults import extra_initialization

__all__ = [
    'DATA_ROOT',
    'CONFIG_ROOT',
    'BINARY_PATH',
    'ARCH',
]

if "XDG_CONFIG_HOME" not in os.environ:
    os.environ["XDG_CONFIG_HOME"] = expanduser('~/.config')
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = expanduser('~/.cache')


def ensure_directory(variable, default):
    path = os.getenv(variable)
    if path is None:
        path = expandvars(default)
    else:
        path = expandvars(expanduser(path))

    # check if expanduser failed:
    if path.startswith('~'):
        path = None
    elif not exists(path):
        os.makedirs(path)
    elif not isdir(path):
        # A file at path already exists
        path = None
    return path


DATA_ROOT = CONFIG_ROOT = None
GRID_MODE = os.getenv('ROOTPY_GRIDMODE') in ('1', 'true')

if (os.getenv('DEBUG', None) or not (GRID_MODE or IN_NOSETESTS)):
    DATA_ROOT = ensure_directory(
        'ROOTPY_DATA', '${XDG_CACHE_HOME}/rootpy')
    CONFIG_ROOT = ensure_directory(
        'ROOTPY_CONFIG', '${XDG_CONFIG_HOME}/rootpy')

if DATA_ROOT is None:
    log.info("Placing user data in /tmp.")
    log.warning(
        "Make sure '~/.cache/rootpy' or $ROOTPY_DATA is a writable "
        "directory so that it isn't necessary to recreate all user "
        "data each time")

    DATA_ROOT = tempfile.mkdtemp()

    @atexit.register
    def __cleanup():
        import shutil
        shutil.rmtree(DATA_ROOT)

BINARY_PATH = None

ARCH = "{0}-{1}".format(machine(), QROOT.gROOT.GetVersionInt())
if BINARY_PATH is None:
    BINARY_PATH = pjoin(DATA_ROOT, ARCH)


@extra_initialization
def show_binary_path():
    log.debug("Using binary path: {0}".format(BINARY_PATH))
