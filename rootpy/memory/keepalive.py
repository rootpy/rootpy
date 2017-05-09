# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import sys
import weakref
import os

from . import log; log = log[__name__]

__all__ = [
    'keepalive',
]

KEEPALIVE = weakref.WeakKeyDictionary()
DISABLED = 'NO_ROOTPY_KEEPALIVE' in os.environ


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except:
        return False
    return True


def keepalive(nurse, *patients):
    """
    Keep ``patients`` alive at least as long as ``nurse`` is around using a
    ``WeakKeyDictionary``.
    """
    if DISABLED:
        return
    if hashable(nurse):
        hashable_patients = []
        for p in patients:
            if hashable(p):
                log.debug("Keeping {0} alive for lifetime of {1}".format(p, nurse))
                hashable_patients.append(p)
            else:
                log.warning("Unable to keep unhashable object {0} "
                            "alive for lifetime of {1}".format(p, nurse))
        KEEPALIVE.setdefault(nurse, set()).update(hashable_patients)
    else:
        log.warning("Unable to keep objects alive for lifetime of "
                    "unhashable object {0}".format(nurse))
