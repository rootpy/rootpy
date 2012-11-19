Capturing ROOT Errors with :py:mod:`rootpy`'s logger
=================================================

More detail about ways to use logging can be found in the rootpy logger module,
see :py:mod:`rootpy.logger`.

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
