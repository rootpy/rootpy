import ctypes as C
import os

from functools import wraps

import ROOT
# This one is here because it doesn't trigger finalSetup()
ROOT.PyConfig.IgnoreCommandLineOptions = True

from . import log
from .logger import set_error_handler, python_logging_error_handler
from .logger.magic import DANGER, fix_ipython_startup

# See magic module for more details
DANGER.enabled = True

if not log["/"].have_handlers():
    # The root logger doesn't have any handlers.
    # Therefore, the application hasn't specified any behaviour, and rootpy
    # uses maximum verbosity.
    log["/"].setLevel(log.NOTSET)

# Show python backtrace if there is a segfault
log["/ROOT.TUnixSystem.DispatchSignals"].showstack(min_level=log.ERROR)

orig_error_handler = set_error_handler(python_logging_error_handler)

DICTS_PATH = MODS_PATH = None

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
    
    # Need to do it again here, since it is overridden by ROOT.
    set_error_handler(python_logging_error_handler)
    
    ROOT.TH1.SetDefaultSumw2(True)
    
    ROOT.gErrorIgnoreLevel = 0
    
    this_dll = C.CDLL(None)
    try:
        EnableAutoDictionary = C.c_int.in_dll(this_dll, "G__EnableAutoDictionary")
    except ValueError:
        pass
    else:
        # Disable automatic dictionary generation
        EnableAutoDictionary.value = 0
    
    # TODO(pwaller): idea, `execfile("userdata/initrc.py")` here?
    #                 note: that wouldn't allow the user to override the default
    #                       canvas size, for example.
    
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
    # trigger our default options then, and .
    
    finalSetup = ROOT.__class__._ModuleFacade__finalSetup
    @wraps(finalSetup)
    def wrapFinalSetup(*args, **kwargs):
        
        if os.environ.get("ROOTPY_DEBUG", None) and rp_module_level_in_stack():
            # Check to see if we're at module level anywhere in rootpy.
            # If so, that's not ideal.
            l = log["bug"]
            l.showstack()
            l.debug("finalSetup triggered from rootpy at module-level. "
                    "Please report this.")
        
        # if running in the ATLAS environment suppress a known harmless warning
        if os.environ.get("AtlasVersion", None):
            regex = "^duplicate entry .* vectorbool.dll> for level 0; ignored$"
            c = log["/ROOT.TEnvRec.ChangeValue"].ignore(regex)
            with c:
                result = finalSetup(*args, **kwargs)
        else:
            result = finalSetup(*args, **kwargs)
        
        configure_defaults()
        
        return result
    
    ROOT.__class__._ModuleFacade__finalSetup = wrapFinalSetup
    
    if "__IPYTHON__" in __builtins__:
        # ROOT has a bug causing it to print (Bool_t)1 to the console.
        fix_ipython_startup(finalSetup)
    
else:
    initialized = True
    configure_defaults()

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700
