"""
Provide some features for writing scripts which provide interactive features

wait: Keeps root alive until ctrl-c is pressed or all canvases are closed
"""

from .rootwait import wait_for_zero_canvases, wait
