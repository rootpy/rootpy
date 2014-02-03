# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Provide some features for writing scripts which provide interactive features

wait: Keeps root alive until ctrl-c is pressed or all canvases are closed
interact: Starts a python console at the call site
"""
from __future__ import absolute_import

from .. import log; log = log[__name__]
from .console import interact
from .rootwait import wait, wait_for_zero_canvases, wait_for_browser_close
from .notebook import configure as configure_notebook

__all__ = [
    'interact',
    'wait', 'wait_for_zero_canvases', 'wait_for_browser_close',
    'configure_notebook',
]
