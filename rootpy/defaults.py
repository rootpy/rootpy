# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import sys
import ctypes as C
import os
from functools import wraps

import ROOT
# This doesn't trigger finalSetup()
ROOT.PyConfig.IgnoreCommandLineOptions = True

from . import log; log = log[__name__]
from . import QROOT, IN_NOSETESTS
from .logger import set_error_handler, python_logging_error_handler
from .logger.magic import DANGER, fix_ipython_startup


__all__ = []


if not log["/"].have_handlers():
    # The root logger doesn't have any handlers.
    # Therefore, the application hasn't specified any behaviour, and rootpy
    # uses maximum verbosity.
    log["/"].setLevel(log.NOTSET)

use_rootpy_handler = not os.environ.get('NO_ROOTPY_HANDLER', False)
# rootpy's logger magic is not safe in Python 3, yet
use_rootpy_magic = not os.environ.get('NO_ROOTPY_MAGIC', False) and sys.version_info[0] < 3

if use_rootpy_handler:
    if use_rootpy_magic:
        # See magic module for more details
        DANGER.enabled = True
    else:
        log.debug('logger magic disabled')
        DANGER.enabled = False
    # Show python backtrace if there is a segfault
    log["/ROOT.TUnixSystem.DispatchSignals"].show_stack(min_level=log.ERROR)
    orig_error_handler = set_error_handler(python_logging_error_handler)
else:
    log.debug('ROOT error handler disabled')

DICTS_PATH = MODS_PATH = None

# Activate the storage of the sum of squares of errors by default.
QROOT.TH1.SetDefaultSumw2(True)
# Activate use of underflows and overflows in `Fill()` in the
# computation of statistics (mean value, RMS) by default.
QROOT.TH1.StatOverflows(True)
# Setting the above static parameters below in the configure_defaults function
# may be too late. For example, the first histogram will be inited before these
# are set.

_initializations = []


def extra_initialization(fn):
    """
    Function decorator which adds `fn` to the list of functions to be called
    at some point after ROOT has been initialized.
    """
    if initialized:
        fn()
    else:
        _initializations.append(fn)
    return fn


def configure_defaults():
    """
    This function is executed immediately after ROOT's finalSetup
    """
    log.debug("configure_defaults()")

    global initialized
    initialized = True

    if use_rootpy_handler:
        # Need to do it again here, since it is overridden by ROOT.
        set_error_handler(python_logging_error_handler)

    if os.environ.get('ROOTPY_BATCH', False) or IN_NOSETESTS:
        ROOT.gROOT.SetBatch(True)
        log.debug('ROOT is running in batch mode')

    ROOT.gErrorIgnoreLevel = 0

    this_dll = C.CDLL(None)
    try:
        EnableAutoDictionary = C.c_int.in_dll(
            this_dll, "G__EnableAutoDictionary")
    except ValueError:
        pass
    else:
        # Disable automatic dictionary generation
        EnableAutoDictionary.value = 0

    # TODO(pwaller): idea, `execfile("userdata/initrc.py")` here?
    #                note: that wouldn't allow the user to override the default
    #                      canvas size, for example.

    for init in _initializations:
        init()


def rp_module_level_in_stack():
    """
    Returns true if we're during a rootpy import
    """
    from traceback import extract_stack
    from rootpy import _ROOTPY_SOURCE_PATH
    modlevel_files = [filename for filename, _, func, _ in extract_stack()
                      if func == "<module>"]

    return any(path.startswith(_ROOTPY_SOURCE_PATH) for path in modlevel_files)


# Check in case the horse has already bolted.
# If initialization has already taken place, we can't wrap it.
if hasattr(ROOT.__class__, "_ModuleFacade__finalSetup"):
    initialized = False

    # Inject our own wrapper in place of ROOT's finalSetup so that we can
    # trigger our default options then.

    finalSetup = ROOT.__class__._ModuleFacade__finalSetup

    @wraps(finalSetup)
    def wrapFinalSetup(*args, **kwargs):

        log.debug("PyROOT's finalSetup() has been triggered")

        if os.environ.get("ROOTPY_DEBUG", None) and rp_module_level_in_stack():
            # Check to see if we're at module level anywhere in rootpy.
            # If so, that's not ideal.
            l = log["bug"]
            l.show_stack()
            l.debug("PyROOT's finalSetup() triggered from rootpy at "
                    "module-level. Please report this.")

        # if running in the ATLAS environment suppress a known harmless warning
        if os.environ.get("AtlasVersion", None):
            regex = "^duplicate entry .* vectorbool.dll> for level 0; ignored$"
            c = log["/ROOT.TEnvRec.ChangeValue"].ignore(regex)
            with c:
                result = finalSetup(*args, **kwargs)
        else:
            result = finalSetup(*args, **kwargs)

        log.debug(
            "PyROOT's finalSetup() has been called "
            "(gROOT.IsBatch()=={0})".format(ROOT.gROOT.IsBatch()))

        configure_defaults()

        return result

    wrapFinalSetup._orig_func = finalSetup

    ROOT.__class__._ModuleFacade__finalSetup = wrapFinalSetup

    if '__IPYTHON__' in __builtins__:
        # ROOT has a bug causing it to print (Bool_t)1 to the console.
        fix_ipython_startup(finalSetup)

else:
    initialized = True
    configure_defaults()
