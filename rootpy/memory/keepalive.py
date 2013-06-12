# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import weakref

from . import log; log = log[__name__]

KEEPALIVE = weakref.WeakKeyDictionary()

def keepalive(nurse, *patients):
    """
    Keep ``patients`` alive at least as long as ``nurse`` is around using a
    ``WeakKeyDictionary``.
    """
    for p in patients:
        log.debug("Keeping {0} alive for {1} lifetime".format(p, nurse))
    KEEPALIVE.setdefault(nurse, set()).update(patients)
