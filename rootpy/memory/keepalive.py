# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import sys
import weakref
import os
from collections import Hashable

from . import log; log = log[__name__]

__all__ = [
    'keepalive',
]

KEEPALIVE = weakref.WeakKeyDictionary()
DISABLED = 'NO_ROOTPY_KEEPALIVE' in os.environ


def keepalive(nurse, *patients):
    """
    Keep ``patients`` alive at least as long as ``nurse`` is around using a
    ``WeakKeyDictionary``.
    """
    if DISABLED:
        return
    for p in patients:
        log.debug("Keeping {0} alive for lifetime of {1}".format(p, nurse))
    if sys.version_info[0] >= 3 and not isinstance(nurse, Hashable):
        # PyROOT missing __hash__ for Python 3
        nurse.__class__.__hash__ = object.__hash__
    KEEPALIVE.setdefault(nurse, set()).update(patients)
