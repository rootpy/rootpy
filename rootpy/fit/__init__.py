# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]
from .. import QROOT
from .fit import fit
from .workspace import *


def mute_roostats():
    # suppress RooStats' rather verbose INFO messages
    log.debug("suppressing RooStats messages below the WARNING level")
    QROOT.RooMsgService.instance().setGlobalKillBelow(QROOT.RooFit.WARNING)
