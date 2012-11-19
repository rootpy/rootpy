import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.TH1.SetDefaultSumw2(True)

# Cause ROOT to abort (and give a stack trace) if it encounters an error, rather
# than continuing along blindly.
ROOT.gErrorAbortLevel = ROOT.kError
ROOT.gErrorIgnoreLevel = 0

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700

from .logger import set_error_handler, python_logging_error_handler

# Needed to avoid spurious ROOT warnings
# WARNING:ROOT.TGClient.GetFontByName] couldn't retrieve font -*-helvetica-medium-r-*-*-12-*-*-*-*-*-iso8859-1, using "fixed"
# WARNING:ROOT.TGClient.GetFontByName] couldn't retrieve font -*-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1, using "fixed"
# WARNING:ROOT.TGClient.GetFontByName] couldn't retrieve font -*-courier-medium-r-*-*-12-*-*-*-*-*-iso8859-1, using "fixed"
# WARNING:ROOT.TGClient.GetFontByName] couldn't retrieve font -*-helvetica-medium-r-*-*-10-*-*-*-*-*-iso8859-1, using "fixed"
ROOT.TCanvas

orig_error_handler = set_error_handler(python_logging_error_handler)
