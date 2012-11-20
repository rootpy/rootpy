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

def configure_defaults():
    ROOT.TH1.SetDefaultSumw2(True)

    # Cause ROOT to abort (and give a stack trace) if it encounters an error,
    # rather than continuing along blindly.
    ROOT.gErrorAbortLevel = ROOT.kError
    ROOT.gErrorIgnoreLevel = 0

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
    
    finalSetup = ROOT.__class__._ModuleFacade__finalSetup
    @wraps(finalSetup)
    def wrapFinalSetup(*args, **kwargs):
        
        if os.environ.get("ROOTPY_DEBUG", None) and rp_module_level_in_stack():
            l = log["bug"]
            l.showstack()
            l.debug("rootpy import isn't complete. Please report this.")
        
        if os.environ.get("AtlasVersion", None):
            regex = "^duplicate entry .* for level 0; ignored$"
            c = log["/ROOT.TEnvRec.ChangeValue"].ignore(regex)
        else:
            @contextmanager
            def c(): yield
        
        with c:
            result = finalSetup(*args, **kwargs)
        
        configure_defaults()
        return result
    
    ROOT.__class__._ModuleFacade__finalSetup = wrapFinalSetup
    
    fix_ipython_startup(finalSetup)
    
else:
    configure_defaults()

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700
