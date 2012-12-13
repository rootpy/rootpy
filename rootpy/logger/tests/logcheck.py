# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import logging
import re

from functools import wraps

import rootpy

class LogCapture(logging.Handler):
    def __init__(self, logger):
        logging.Handler.__init__(self)
        self.records = []
        self.logger = logger

    def __enter__(self):
        self.logger.addHandler(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self)

    def emit(self, record):
        self.records.append(record)

    def contains(self, level, message_re):
        return any(r.levelname == level and message_re.search(r.getMessage())
                   for r in self.records)

class EnsureLogContains(object):
    def __init__(self, level, message_pattern):
        self.level = level
        self.message_pattern = message_pattern
        self.msg_re = re.compile(message_pattern)

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):

            with LogCapture(rootpy.log["/ROOT"]) as captured:
                try:
                    return func(*args, **kwargs)
                finally:
                    assert captured.contains(self.level, self.msg_re), (
                        "Expected `{0}` to emit a {1} message matching '{2}'. "
                        "It did not."
                        .format(func.__name__, self.level, self.message_pattern)
                    )

        return wrapped