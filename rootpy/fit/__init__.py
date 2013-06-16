# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]

# suppress RooStats' rather verbose INFO messages by default
from .. import QROOT
QROOT.RooMsgService.instance().setGlobalKillBelow(QROOT.RooFit.WARNING)


from .fit import fit
from .workspace import *
