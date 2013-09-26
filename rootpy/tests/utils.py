# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from . import log; log = log[__name__]
from .. import _get_class, INIT_REGISTRY_ROOTPY

__all__ = [
    'iter_rootpy_classes',
]


def iter_rootpy_classes():
    for name, path in INIT_REGISTRY_ROOTPY.items():
        try:
            cls = _get_class(path, name)
        except:
            log.warning(
                "unable to get class {0} at {1}".format(name, path))
            continue
        else:
            yield cls
