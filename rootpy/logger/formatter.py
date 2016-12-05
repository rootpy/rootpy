# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Provides a ``CustomFormatter`` and ``CustomColoredFormatter`` which are enable
to insert ANSI color codes.
"""
from __future__ import absolute_import

import logging

__all__ = [
    'CustomFormatter',
    'CustomColoredFormatter',
]

# The background is set with 40 plus the number of the color, and the foreground with 30
RED, YELLOW, BLUE, WHITE = 1, 3, 4, 7

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
FORMAT = "{color}{levelname}$RESET:$BOLD{name}$RESET] {message}"

def insert_seqs(message):
    return message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)

def remove_seqs(message):
    return message.replace("$RESET", "").replace("$BOLD", "")

COLORS = {
    'DEBUG'      : BLUE,
    'INFO'       : WHITE,
    'WARNING'    : YELLOW,
    'ERROR'      : RED,
    'CRITICAL'   : RED,
}


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=remove_seqs(FORMAT), datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        if not hasattr(record, "message"):
            record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        return self._fmt.format(color="", **record.__dict__)


class CustomColoredFormatter(CustomFormatter):
    def __init__(self, fmt=insert_seqs(FORMAT), datefmt=None, use_color=True):
        CustomFormatter.__init__(self, fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            record.color = COLOR_SEQ % (30 + COLORS[levelname])
        else:
            record.color = ""
        if not hasattr(record, "message"):
            record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        return self._fmt.format(**record.__dict__)
