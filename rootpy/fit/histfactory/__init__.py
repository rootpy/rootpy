# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]
from ... import ROOT_VERSION

if ROOT_VERSION >= 53404:
    from .histfactory import *
    from .utils import *
else:
    import warnings
    warnings.warn('histfactory requires ROOT 5.34.04 but you are using %d' %
                  ROOT_VERSION)
