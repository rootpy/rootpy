
.. _logging:

=======
Logging
=======

.. currentmodule:: rootpy.logger

More detail about ways to use logging can be found in the rootpy logger module,
see :py:mod:`rootpy.logger`.


Python logging basics
=====================

In Python's standard library there is a ``logging`` module. It's good to make
use of this, because it means that you can interoperate with a large range of
tools which already use it. :py:mod:`rootpy` extends the default logger to add
a range of useful utilities, but underlying it is the normal logging behaviour.

Python's ``logging`` module allows you to define a hierarchy of loggers.
Confusingly, the base of the hierarchy is called ``root``. Each logger in a 
hierarchy has a ``.level`` (one of ``CRITICAL``, ``ERROR``, ``WARNING``,
``INFO``, ``DEBUG``, ``NOTSET``), which defines the minimum severity of messages
which will be passed onto the handlers.

Handlers can be attached to specific loggers, or anywhere in the hierarchy. For
example, a handler attached to the logger ``logging.root`` would recieve all
messages which meet the ``root`` logger's minimum level.

Handlers have a ``level``, independently of a logger. This means that a handler
can be configured to only receive certain messages.

By default, Python configures no handlers. This means that ordinarily, after
emitting a brief one-time warning about no handlers being configured, messages
get sent into the void, never to be heard from again. In addition, the default
state of the ``logging.root`` logger is ``WARNING``, so at first, no messages
less severe than that (``INFO`` and ``DEBUG``) will be processed.

This makes it reasonable to leave ``log.debug`` statements in non-performance
critical code, since by default they aren't shown.


Capturing ROOT messages with :py:mod:`rootpy.logger`
====================================================

rootpy provides a mechanism to capture ROOT's log messages and if appropriate,
raise an error. This can be a big boon for debugging and makes it possible to
do things which were not previously possible, such as capture messages to show
them where they are needed (think of web services, for example).

As a trivial example, it means that this:

.. sourcecode:: ipython

	In [2]: r = ROOT.TFile("test.root")
	Error in <TFile::TFile>: file test.root does not exist

	In [3]: r.myanalysisdata
	---------------------------------------------------------------------------
	AttributeError                            Traceback (most recent call last)
	<ipython-input-3-c06f1156de1b> in <module>()
	----> 1 r.myanalysisdata

	AttributeError: TFile object has no attribute 'myanalysisdata'

Now does this:

.. sourcecode:: ipython

	In [3]: import rootpy
	In [4]: f = ROOT.TFile("test.root")
	> Warning: No logger for 'ROOT', adding a default
	>          Suppress with 'import logging; logging.basicConfig()'
	ERROR:ROOT.TFile.TFile:file test.root does not exist
	---------------------------------------------------------------------------
	ROOTError                                 Traceback (most recent call last)
	<ipython-input-4-e2c494f4c26e> in <module>()
	----> 1 f = ROOT.TFile("test.root")

	<ipython-input-4-e2c494f4c26e> in <module>()
	----> 1 f = ROOT.TFile("test.root")

	.../rootpy/logger/roothandler.py in python_logging_error_handler(level, abort, location, msg)
	     50                         # We can't raise an exception from here because ctypes/PyROOT swallows it.
	     51                         # Hence the need for dark magic, we re-raise it within a trace.
	---> 52                         raise ROOTError(level, location, msg)
	     53                 except RuntimeError:
	     54                         _, exc, traceback = sys.exc_info()

	ROOTError: level=3000, loc='TFile::TFile', msg='file test.root does not exist'

This means that the code fails early, even better it's at the point of the
problem. It uses python's logging mechanism, so now it's possible to filter
messages with ease.

.. warning::

	The astute amongst you might have noticed a passing mention of dark magic. This
	feature is still a little experimental, and is currently limited to CPython 2.6-2.7.

.. note::

	If errors don't give you backtraces as you expect, it might be necessary to:

	.. sourcecode:: python

		import rootpy.logger.magic as M; M.DANGER.enabled = True

	.. and if that doesn't work, file a bug.

The first thing to note is the warning mentioning a lack of a default logger. To
suppress this, configure logging any way you please, or use the snippet from the
warning itself.

.. sourcecode:: python

	import logging
	# Most verbose log level
	logging.basicConfig(level=logging.DEBUG)

Once that is done, you can use python's normal logging API to suppress or
highlight log messages coming from particular places with ease:

.. sourcecode:: python
	
	import logging
	# Suppress "debug"-level notices from TCanvas that it has saved a .png
	logging.getLogger("ROOT.TCanvas").setLevel(logging.WARNING)

If you want to get the stack trace at the point where any ROOT message is coming
from, you can decrease the abort level to the minimum:

.. sourcecode:: python

	import ROOT
	ROOT.gErrorAbortLevel = 0


What else does rootpy's logging do for me?
==========================================

:py:mod:`rootpy.logger` adds a check to ensure that a handler is configured
against the ``rootpy`` logging namespace and, if not, installs a default one.

There is some syntactic sugar to obtain loggers in a given namespace:

.. sourcecode:: python

	import rootpy

	# logger in the rootpy logging namespace
   # (whose parent is python's `logging.root`)
   log = rootpy.log

	log["child"] # logger in the rootpy.child namespace
	log["/ROOT"] # ROOT
	log["/ROOT.TFile"] # ROOT.TFile

:py:mod:`rootpy.logger` can also help you identify *where* messages are coming
from, using :meth:`rootpy.logger.extended_logger.ExtendedLogger.showstack`.

.. sourcecode:: python

	log["/"].setLevel(log.NOTSET)
	# Show stack traces for
	log["/ROOT"].showstack()
