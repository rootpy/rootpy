# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import log; log = log[__name__]
from .. import QROOT

try:
    QROOT.RooFit
    QROOT.RooMsgService

except:
    import warnings
    warnings.warn(
        "rootpy.stats requires libRooFit and libRooStats. "
        "Please recompile ROOT with --enable-roofit")
    __all__ = []

else:
    import os
    from .workspace import Workspace

    __all__ = [
        'mute_roostats',
        'Workspace',
    ]


    def mute_roostats():
        # suppress RooStats' rather verbose INFO messages unless DEBUG is set
        if not os.environ.get('DEBUG', False):
            log.debug("suppressing RooStats messages below the WARNING level")
            QROOT.RooMsgService.instance().setGlobalKillBelow(QROOT.RooFit.WARNING)
