"""
This module exists for backwards compatibility
"""
import warnings

warnings.warn("rootpy.hep is deprecated. Use rootpy.extern.hep instead.",
        DeprecationWarning, stacklevel=2)

from .extern.hep import pdg
