from .. import log; log = log[__name__]

import ROOT
ROOT.gROOT.SetBatch(True)

from .student import Student
from .supervisor import Supervisor
