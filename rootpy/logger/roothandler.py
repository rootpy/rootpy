# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ctypes
import logging
import re
import sys

from . import root_logger, log
from .magic import DANGER, set_error_handler, re_execute_with_exception

__all__ = [
    'fixup_msg',
    'python_logging_error_handler',
]


class SHOWTRACE:
    enabled = False

SANE_REGEX = re.compile("^[^\x80-\xFF]*$")


class Initialized:
    value = False

ABORT_LEVEL = log.ERROR


def fixup_msg(lvl, msg):

    # Fixup for this ERROR to a WARNING because it has a reasonable fallback.
    # WARNING:ROOT.TGClient.TGClient] can't open display "localhost:10.0", switching to batch mode...
    #  In case you run from a remote ssh session, reconnect with ssh -Y
    if "switching to batch mode..." in msg and lvl == logging.ERROR:
        return logging.WARNING, msg

    return lvl, msg


def python_logging_error_handler(level, root_says_abort, location, msg):
    """
    A python error handler for ROOT which maps ROOT's errors and warnings on
    to python's.
    """
    from ..utils import quickroot as QROOT

    if not Initialized.value:
        try:
            QROOT.kTRUE
        except AttributeError:
            # Python is exiting. Do nothing.
            return
        QROOT.kInfo, QROOT.kWarning, QROOT.kError, QROOT.kFatal, QROOT.kSysError
        QROOT.gErrorIgnoreLevel
        Initialized.value = True

    try:
        QROOT.kTRUE
    except RuntimeError:
        # Note: If the above causes us problems, it's because this logging
        #       handler has been called multiple times already with an
        #       exception. In that case we need to force upstream to raise it.
        _, exc, traceback = sys.exc_info()
        caller = sys._getframe(2)
        re_execute_with_exception(caller, exc, traceback)

    if level < QROOT.gErrorIgnoreLevel:
        # Needed to silence some "normal" startup warnings
        # (copied from PyROOT Utility.cxx)
        return

    if sys.version_info[0] >= 3:
        location = location.decode('utf-8')
        msg = msg.decode('utf-8')

    log = root_logger.getChild(location.replace("::", "."))

    if level >= QROOT.kSysError or level >= QROOT.kFatal:
        lvl = logging.CRITICAL
    elif level >= QROOT.kError:
        lvl = logging.ERROR
    elif level >= QROOT.kWarning:
        lvl = logging.WARNING
    elif level >= QROOT.kInfo:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG

    if not SANE_REGEX.match(msg):
        # Not ASCII characters. Escape them.
        msg = repr(msg)[1:-1]

    # Apply fixups to improve consistency of errors/warnings
    lvl, msg = fixup_msg(lvl, msg)

    log.log(lvl, msg)

    # String checks are used because we need a way of (un)forcing abort without
    # modifying a global variable (gErrorAbortLevel) for the multithread tests
    abort = lvl >= ABORT_LEVEL or "rootpy.ALWAYSABORT" in msg or root_says_abort
    if abort and not "rootpy.NEVERABORT" in msg:
        caller = sys._getframe(1)

        try:
            # We can't raise an exception from here because ctypes/PyROOT swallows it.
            # Hence the need for dark magic, we re-raise it within a trace.
            from .. import ROOTError
            raise ROOTError(level, location, msg)
        except RuntimeError:
            _, exc, traceback = sys.exc_info()

        if SHOWTRACE.enabled:
            from traceback import print_stack
            print_stack(caller)

        if DANGER.enabled:
            # Avert your eyes, dark magic be within...
            re_execute_with_exception(caller, exc, traceback)

    if root_says_abort:
        log.CRITICAL("abort().. expect a stack trace")
        ctypes.CDLL(None).abort()
