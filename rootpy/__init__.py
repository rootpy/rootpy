from .logger import log

from . import defaults
from .info import __version_info__, __version__

import warnings

# show deprecation warnings
warnings.filterwarnings('always', category=DeprecationWarning)

class ROOTError(RuntimeError):
    """
    Exception class representing a ROOT error/warning message.
    """
    def __init__(self, level, location, msg):
        self.level, self.location, self.msg = level, location, msg

    def __str__(self):
        return "level={0}, loc='{1}', msg='{2}'".format(
            self.level, self.location, self.msg)

def rootpy_source_dir():
    import rootpy
    from os.path import abspath, dirname
    from inspect import getfile
    from sys import modules
    path = dirname(getfile(modules[__name__]))
    absp = abspath(path)
    return path, absp

_ROOTPY_SOURCE_PATH, _ROOTPY_SOURCE_ABSPATH = rootpy_source_dir()
del rootpy_source_dir

