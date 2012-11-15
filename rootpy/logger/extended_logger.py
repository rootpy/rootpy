import logging

LoggerClass = logging.getLoggerClass()
class ExtendedLogger(LoggerClass):
    """
    A logger class which provides a few niceties, including automatically
    enabling logging if no handlers are available.
    """

    def __init__(self, name):
        LoggerClass.__init__(self, name)
        self.__dict__.update(logging._levelNames)
    
    def trace(self, level=logging.DEBUG, show_enter=True, show_exit=True):
        """
        Show function entry and exit with values, defaults to debug log level.
        """
        from rootpy.logger import log_trace
        return log_trace(self, level, show_enter, show_exit)

    def basic_config_colorized(self):
        """
        Configure logging with a coloured output
        """
        from rootpy.logger.color import default_log_handler
        default_log_handler()

    def have_handlers(self):
        logger = self
        while logger:
            if logger.handlers:
                return True
            logger = logger.parent
        return False

    def _log(self, lvl, *args, **kwargs):
        if self.isEnabledFor(lvl) and not self.have_handlers():
            self.basic_config_colorized()

            l = self.getLogger("rootpy.logger")
            l.info("| No default log handler configured. See `logging` module |")
            l.info(" \   To suppress: 'rootpy.log.basic_config_colorized()   /")

        return LoggerClass._log(self, lvl, *args, **kwargs)

    def getLogger(self, name):
        if name == "/":
            name = None
        return logging.getLogger(name)

    def __getitem__(self, suffix):
        """
        Provides ``log["child"]`` syntax to obtain a child logger, or
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

logging.setLoggerClass(ExtendedLogger)