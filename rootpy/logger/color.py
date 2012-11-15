"""
Provides a ``CustomFormatter`` and ``CustomColoredFormatter`` which are enable
to insert ANSI color codes.
"""

import os
import logging

FORCE_COLOR = False

# The background is set with 40 plus the number of the color, and the foreground with 30
RED, YELLOW, BLUE, WHITE = 1, 3, 4, 7

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
FORMAT = "%(levelname)s:$BOLD%(name)s$RESET] %(message)s"

def insert_seqs(message):
    return message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)

def remove_seqs(message):
    return message.replace("$RESET", "").replace("$BOLD", "")

COLORS = {
    'DEBUG'   : BLUE,
    'INFO'    : WHITE,
    'WARNING' : YELLOW,
    'ERROR'   : RED,
    'CRITICAL'   : RED,
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return logging.Formatter.format(self, record)
        
class CustomColoredFormatter(CustomFormatter):
    def __init__(self, msg, datefmt=None, use_color=True):
        msg = insert_seqs(msg)
        logging.Formatter.__init__(self, msg, datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            color_seq = COLOR_SEQ % (30 + COLORS[levelname])
            record.levelname = color_seq + levelname + RESET_SEQ
        return logging.Formatter.format(self, record)

def default_log_handler(level=logging.DEBUG, singleton={}):
    """
    Instantiates a default log handler, with colour if we're connected to a
    terminal.
    """
    if "value" in singleton:
        return singleton["value"]
        
    handler = logging.StreamHandler()
    if os.isatty(handler.stream.fileno()) or FORCE_COLOR:
        handler.setFormatter(CustomColoredFormatter(insert_seqs(FORMAT)))
    else:
        handler.setFormatter(CustomFormatter(remove_seqs(FORMAT)))
    
    # Make the top level logger and make it as verbose as possible.
    # The log messages which make it to the screen are controlled by the handler
    log = logging.getLogger()
    log.addHandler(handler)
    log.setLevel(level)

    singleton["value"] = handler
    return handler

