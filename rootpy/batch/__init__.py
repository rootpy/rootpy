# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import log; log = log[__name__]

import ROOT
ROOT.gROOT.SetBatch(True)

from .student import Student
from .supervisor import Supervisor
