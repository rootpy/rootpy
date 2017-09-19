.. rootpy.ROOT

Simple Integration into Existing Applications
=============================================

The :py:mod:`rootpy.ROOT` module is intended to be a drop-in replacement for
ordinary PyROOT imports by mimicking PyROOT's interface. Both ROOT and rootpy
classes can be accessed in a harmonized way through this module. This means you
can take advantage of rootpy classes automatically by replacing ``import ROOT``
with ``import rootpy.ROOT as ROOT`` or ``from rootpy import ROOT`` in your code,
while maintaining backward compatibility with existing use of ROOT's classes.

Under :py:mod:`rootpy.ROOT`, classes are automatically "asrootpy'd" *after* the
ROOT constructor has been called:

.. sourcecode:: python

    >>> import rootpy.ROOT as ROOT
    >>> ROOT.TH1F('name', 'title', 10, 0, 1)
    Hist('name')

Access rootpy classes under :py:mod:`rootpy.ROOT` without needing to remember
where to import them from in rootpy:

.. sourcecode:: python

    >>> import rootpy.ROOT as ROOT
    >>> ROOT.Hist(10, 0, 1, name='name', type='F')
    Hist('name')

