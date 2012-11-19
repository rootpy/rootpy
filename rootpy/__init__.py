from . import defaults
from .info import __version_info__, __version__
from .logger import log

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
