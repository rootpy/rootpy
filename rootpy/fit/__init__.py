# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os

from .. import log; log = log[__name__]
from .. import QROOT
from .fit import nll_fit, minimize
from .workspace import Workspace

__all__ = [
    'nll_fit', 'minimize',
    'Workspace',
    'mute_roostats',
]

def mute_roostats():
    # suppress RooStats' rather verbose INFO messages unless DEBUG is set
    if not os.environ.get('DEBUG', False):
        log.debug("suppressing RooStats messages below the WARNING level")
        QROOT.RooMsgService.instance().setGlobalKillBelow(QROOT.RooFit.WARNING)
