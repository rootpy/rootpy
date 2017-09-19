from __future__ import absolute_import

from .. import log; log = log[__name__]
from .. import QROOT, IN_NOSETESTS, ROOTError
from ..context import do_nothing
from ..utils.silence import silence_sout

try:
    context = silence_sout if IN_NOSETESTS else do_nothing
    with context():
        QROOT.RooFit
        QROOT.RooMsgService

except (AttributeError, ROOTError):
    import warnings
    warnings.warn(
        "rootpy.stats requires libRooFit and libRooStats. "
        "Please recompile ROOT with roofit enabled")
    __all__ = []

else:
    import os
    from .. import stl

    # generate dictionaries
    stl.stack('RooAbsArg*,deque<RooAbsArg*>',
              headers='<stack>;<deque>;RooRealVar.h')

    from .workspace import Workspace
    from .modelconfig import ModelConfig
    from .collection import ArgSet, ArgList
    from .value import RealVar
    from .pdf import Simultaneous, AddPdf, ProdPdf

    __all__ = [
        'mute_roostats',
        'Workspace',
        'ModelConfig',
        'ArgSet',
        'ArgList',
        'RealVar',
        'Simultaneous',
        'AddPdf',
        'ProdPdf',
    ]


    def mute_roostats():
        """
        suppress RooStats' rather verbose INFO messages unless DEBUG is set
        """
        if not os.environ.get('DEBUG', False):
            log.debug("suppressing RooStats messages below the WARNING level")
            QROOT.RooMsgService.instance().setGlobalKillBelow(
                QROOT.RooFit.WARNING)
