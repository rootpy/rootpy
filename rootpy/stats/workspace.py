# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT
from ..core import NamedObject

__all__ = [
    'Workspace',
]


class Workspace(NamedObject, QROOT.RooWorkspace):

    pass
