"""
This module handles creation of the user-data area
"""
import os
import tempfile
import atexit
        
from os.path import expanduser, expandvars, exists, isdir, join as pjoin

from . import log; log = log[__name__]
from rootpy.defaults import extra_initialization

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
if os.getenv('ROOTPY_GRIDMODE') not in ('1', 'true'):
    DATA_ROOT = ensure_directory('ROOTPY_DATA', '${XDG_CACHE_HOME}/rootpy')
    CONFIG_ROOT = ensure_directory('ROOTPY_CONFIG', '${XDG_CONFIG_HOME}/rootpy')


if DATA_ROOT is None:
    log.info("Placing user data in /tmp.")
    log.warning("Make sure '~/.cache/rootpy' or $ROOTPY_DATA is a writable "
                "directory so that it isn't necessary to recreate all user data"
                " each time")
    
    DATA_ROOT = tempfile.mkdtemp()

    @atexit.register
    def __cleanup():
        import shutil
        shutil.rmtree(DATA_ROOT)

BINARY_PATH = None

@extra_initialization
def set_binary_path():
    from ROOT import gROOT
    from platform import machine
    ARCH = "{0}-{1}".format(machine(), gROOT.GetVersionInt())
    
    global BINARY_PATH
    if BINARY_PATH is None:
        BINARY_PATH = pjoin(DATA_ROOT, ARCH)
    log.debug("Using binary path: {0}".format(BINARY_PATH))
