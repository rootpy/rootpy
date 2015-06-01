# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import logging
import re
import sys
import traceback
import types
import threading

__all__ = [
    'log_stack',
    'ExtendedLogger',
    'RootLoggerWrapper',
]


class ShowingStack(threading.local):
    inside = False

showing_stack = ShowingStack()


def log_stack(logger, level=logging.INFO, limit=None, frame=None):
    """
    Display the current stack on ``logger``.

    This function is designed to be used during emission of log messages, so it
    won't call itself.
    """
    if showing_stack.inside:
        return
    showing_stack.inside = True
    try:
        if frame is None:
            frame = sys._getframe(1)
        stack = "".join(traceback.format_stack(frame, limit))
        for line in (l[2:] for l in stack.split("\n") if l.strip()):
            logger.log(level, line)
    finally:
        showing_stack.inside = False

LoggerClass = logging.getLoggerClass()


class ExtendedLogger(LoggerClass):
    """
    A logger class which provides a few niceties, including automatically
    enabling logging if no handlers are available.
    """
    def __init__(self, name, *args, **kwargs):
        LoggerClass.__init__(self, name, *args, **kwargs)
        self._init(self)

    @staticmethod
    def _init(self):
        if hasattr(self, "shown_stack_frames"):
            # Don't double _init the root logger
            return
        if sys.version_info >= (3, 4):
            self.__dict__.update(logging._levelToName)
            self.__dict__.update(logging._nameToLevel)
        else:
            self.__dict__.update(logging._levelNames)
        self.show_stack_regexes = []
        self.shown_stack_frames = set()

    def showdeletion(self, *objects):
        """
        Record a stack trace at the point when an ROOT TObject is deleted
        """
        from ..memory import showdeletion as S
        for o in objects:
            S.monitor_object_cleanup(o)

    def ignore(self, message_regex):
        """
        Gives a context manager which filters out messages exactly matching
        ``message_regex`` on the current filter.

        Example:

        .. sourcecode:: python

            with log["/ROOT"].ignore("^this message is ignored$"):
                ROOT.Warning("location", "this message is ignored")

        """
        from . import LogFilter
        return LogFilter(self, message_regex)

    def trace(self, level=logging.DEBUG, show_enter=True, show_exit=True):
        """
        Functions decorated with this function show function entry and exit with
        values, defaults to debug log level.

        :param level: log severity to use for function tracing
        :param show_enter: log function entry
        :param show_enter: log function exit

        Example use:

        .. sourcecode:: python

            log = rootpy.log["/myapp"]
            @log.trace()
            def salut():
                return

            @log.trace()
            def hello(what):
                salut()
                return "42"

            hello("world")
            # Result:
            #   DEBUG:myapp.trace.hello] > ('world',) {}
            #   DEBUG:myapp.trace.salut]  > () {}
            #   DEBUG:myapp.trace.salut]  < return None [0.00 sec]
            #   DEBUG:myapp.trace.hello] < return 42 [0.00 sec]

        Output:

        .. sourcecode:: none

        """
        from . import log_trace
        return log_trace(self, level, show_enter, show_exit)

    def basic_config_colorized(self):
        """
        Configure logging with a coloured output.
        """
        from .color import default_log_handler
        default_log_handler()

    def have_handlers(self):
        logger = self
        while logger:
            if logger.handlers:
                return True
            logger = logger.parent
        return False

    def show_stack(self, message_regex="^.*$", min_level=logging.DEBUG,
        limit=4096, once=True):
        """
        Enable showing the origin of log messages by dumping a stack trace into
        the ``stack`` logger at the :const:``logging.INFO`` severity.

        :param message_regex: is a full-line regex which the message must
            satisfy in order to trigger stack dump
        :param min_level: the minimum severity the message must have in order to
            trigger the stack dump
        :param limit: Maximum stack depth to show
        :param once: Only show the stack once per unique ``(logger, origin line
            of code)``
        """
        value = re.compile(message_regex), limit, once, min_level
        self.show_stack_regexes.append(value)

    @staticmethod
    def frame_unique(f):
        """
        A tuple representing a value which is unique to a given frame's line of
        execution
        """
        return f.f_code.co_filename, f.f_code.co_name, f.f_lineno

    def show_stack_depth(self, record, frame):
        """
        Compute the maximum stack depth to show requested by any hooks,
        returning -1 if there are none matching, or if we've already emitted
        one for the line of code referred to.
        """
        logger = self

        depths = [-1]
        msg = record.getMessage()

        # For each logger in the hierarchy
        while logger:
            to_match = getattr(logger, "show_stack_regexes", ())
            for regex, depth, once, min_level in to_match:
                if record.levelno < min_level:
                    continue
                if not regex.match(record.msg):
                    continue
                # Only for a given regex, line number and logger
                unique = regex, self.frame_unique(frame), record.name
                if once:
                    if unique in logger.shown_stack_frames:
                        # We've shown this one already.
                        continue
                    # Prevent this stack frame from being shown again
                    logger.shown_stack_frames.add(unique)
                depths.append(depth)
            logger = logger.parent
        return max(depths)

    def maybeShowStack(self, record):
        frame = sys._getframe(5)
        if frame.f_code.co_name == "python_logging_error_handler":
            # Special case, don't show python messsage handler in backtrace
            frame = frame.f_back
        depth = self.show_stack_depth(record, frame)
        if depth > 0:
            log_stack(self["/stack"], record.levelno, limit=depth, frame=frame)

    def callHandlers(self, record):
        if self.isEnabledFor(record.levelno) and not self.have_handlers():
            self.basic_config_colorized()
            l = self.getLogger("rootpy.logger")
            l.debug("Using rootpy's default log handler")
        result = LoggerClass.callHandlers(self, record)
        self.maybeShowStack(record)
        return result

    def getLogger(self, name):
        if not name:
            # The root logger is special, and always has the same class.
            # Therefore, we wrap it here to give it nice methods.
            return RootLoggerWrapper(logging.getLogger())
        return logging.getLogger(name)

    def __getitem__(self, suffix):
        """
        Provides ``log["child"]`` syntactic sugar to obtain a child logger, or
        ``log["/absolute"]`` to get a logger with respect to the root logger.
        """
        if suffix.startswith("/"):
            return self.getLogger(suffix[1:])
        return self.getChild(suffix)

    def getChild(self, suffix):
        """
        Taken from CPython 2.7, modified to remove duplicate prefix and suffixes
        """
        if suffix is None:
            return self
        if self.root is not self:
            if suffix.startswith(self.name + "."):
                # Remove duplicate prefix
                suffix = suffix[len(self.name + "."):]
                suf_parts = suffix.split(".")
                if len(suf_parts) > 1 and suf_parts[-1] == suf_parts[-2]:
                    # If we have a submodule's name equal to the parent's name,
                    # omit it.
                    suffix = ".".join(suf_parts[:-1])
            suffix = '.'.join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def __repr__(self):
        return "<ExtendedLogger {0} at 0x{1:x}>".format(self.name, id(self))


class RootLoggerWrapper(ExtendedLogger):
    """
    Wraps python's ``logging.RootLogger`` with our nicer methods.

    RootLoggerWrapper is obtained through ``log["/"]``
    """
    def __init__(self, root_logger):
        self.__dict__["__root_logger"] = root_logger
        self._init(root_logger)

    def __getattr__(self, key):
        return getattr(self.__dict__["__root_logger"], key)

    def __setattr__(self, key, value):
        return setattr(self.__dict__["__root_logger"], key, value)

    def __repr__(self):
        return "<RootLoggerWrapper {0} at 0x{1:x}>".format(self.name, id(self))

logging.setLoggerClass(ExtendedLogger)
