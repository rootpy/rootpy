# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Provide some features for writing scripts which provide interactive features

wait: Keeps root alive until ctrl-c is pressed or all canvases are closed
interact: Starts a python console at the call site
"""

from .. import log; log = log[__name__]

from .console import interact
from .rootwait import wait_for_zero_canvases, wait, wait_for_browser_close

