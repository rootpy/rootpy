from __future__ import absolute_import

from .. import Style

__all__ = [
    'style',
]

def style(name='DEFAULT'):
    return Style("DEFAULT", "Default Style")
