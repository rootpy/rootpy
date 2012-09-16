from .info import __version_info__, __version__
from . import defaults

import warnings

# show deprecation warnings
warnings.filterwarnings('always', category=DeprecationWarning)
