# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module exists for backwards compatibility
"""
import warnings

warnings.warn("rootpy.hep is deprecated. Use rootpy.extern.hep instead.",
        DeprecationWarning, stacklevel=2)

from .extern.hep import pdg
