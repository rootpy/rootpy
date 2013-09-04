# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT
ROOT.gROOT.SetBatch(True)

from .. import log; log = log[__name__]
from .student import Student
from .supervisor import Supervisor

__all__ = [
    'Student',
    'Supervisor',
]
