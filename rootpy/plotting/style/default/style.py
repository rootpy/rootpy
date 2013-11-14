# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import Style

__all__ = [
    'style',
]

def style(name='DEFAULT'):
    return Style("DEFAULT", "Default Style")
