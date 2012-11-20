import logging
import re
import sys

import ROOT

from . import root_logger
from .magic import DANGER, set_error_handler, re_execute_with_exception

class SHOWTRACE:
    enabled = False

SANE_REGEX = re.compile("^[^\x80-\xFF]*$")

class Initialized:
    value = False

def python_logging_error_handler(level, abort, location, msg):
    """
    A python error handler for ROOT which maps ROOT's errors and warnings on
    to python's.
    """
    if not Initialized.value:
        ROOT.kInfo, ROOT.kWarning, ROOT.kError, ROOT.kFatal, ROOT.kSysError
        ROOT.kTRUE
        ROOT.gErrorIgnoreLevel
        Initialized.value = True
    
    try:
        ROOT.kTRUE
    except RuntimeError:
        # Note: If the above causes us problems, it's because this logging
        #       handler has been called multiple times already with an
        #       exception. In that case we need to force upstream to raise it.
        _, exc, traceback = sys.exc_info()
        caller = sys._getframe(2)
        re_execute_with_exception(caller, exc, traceback)
        
    if level < getattr(ROOT, "gErrorIgnoreLevel", -1):
        # Needed to silence some "normal" startup warnings
        # (copied from PyROOT Utility.cxx)
        return

    log = root_logger.getChild(location.replace("::", "."))

    if level >= ROOT.kSysError or level >= ROOT.kFatal:
        lvl = logging.CRITICAL
    elif level >= ROOT.kError:
        lvl = logging.ERROR
    elif level >= ROOT.kWarning:
        lvl = logging.WARNING
    elif level >= ROOT.kInfo:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG

    if not SANE_REGEX.match(msg):
        # Not ASCII characters. Escape them.
        msg = repr(msg)[1:-1]

    log.log(lvl, msg)

    # String checks are used because we need a way of (un)forcing abort without
    # modifying a global variable (gErrorAbortLevel) for the multithread tests
    if "rootpy.ALWAYSABORT" in msg or abort and not "rootpy.NEVERABORT" in msg:
        caller = sys._getframe(1)

        try:
            # We can't raise an exception from here because ctypes/PyROOT swallows it.
            # Hence the need for dark magic, we re-raise it within a trace.
            from rootpy import ROOTError
            raise ROOTError(level, location, msg)
        except RuntimeError:
            _, exc, traceback = sys.exc_info()

        if SHOWTRACE.enabled:
            from traceback import print_stack
            print_stack(caller)

        if DANGER.enabled:
            # Avert your eyes, dark magic be within...
            re_execute_with_exception(caller, exc, traceback)
