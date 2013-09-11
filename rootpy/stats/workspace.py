# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from . import log; log = log[__name__]
from .. import QROOT
from ..base import NamedObject

__all__ = [
    'Workspace',
]


class Workspace(NamedObject, QROOT.RooWorkspace):

    _ROOT = QROOT.RooWorkspace

    def __call__(self, *args):
        """
        Need to provide an alternative to RooWorkspace::import since import is
        a reserved word in Python and would be a syntax error.
        """
        return getattr(super(Workspace, self), 'import')(*args)
