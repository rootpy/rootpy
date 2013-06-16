# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]
from ... import ROOT_VERSION, ROOTVersion

MIN_ROOT_VERSION = ROOTVersion(53404)

if ROOT_VERSION >= MIN_ROOT_VERSION:
    from .histfactory import *
    from .utils import *
else:
    import warnings
    warnings.warn(
        "histfactory requires ROOT {0} but you are using {1}".format(
            MIN_ROOT_VERSION, ROOT_VERSION))
