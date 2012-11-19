import os

import ROOT

from . import log
from .logger import set_error_handler, python_logging_error_handler
from .logger.magic import DANGER

DANGER.enabled = True

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.TH1.SetDefaultSumw2(True)

# Cause ROOT to abort (and give a stack trace) if it encounters an error, rather
# than continuing along blindly.
ROOT.gErrorAbortLevel = ROOT.kError
ROOT.gErrorIgnoreLevel = 0

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700

orig_error_handler = set_error_handler(python_logging_error_handler)

if not log["/"].have_handlers():
    # The root logger doesn't have any handlers.
    # Therefore, the application hasn't specified any behaviour, and rootpy
    # uses maximum verbosity.
    log["/"].setLevel(log.NOTSET)