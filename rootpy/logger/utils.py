# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os

__all__ = [
    'check_tty',
]


def check_tty(stream):
    if not hasattr(stream, 'fileno'):
        return False
    try:
        fileno = stream.fileno()
        return os.isatty(fileno)
    except (OSError, IOError):
        return False
