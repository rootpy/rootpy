.. _plotting:

========
Plotting
========

.. currentmodule:: rootpy.plotting

rootpy provides a more pythonic interface for histograms, graphs, canvases and
plotting styles.

Histograms
==========

rootpy simplifies ROOT's histogram class hierarchy into three classes:
``Hist``, ``Hist2D``, and ``Hist3D``. These classes dynamically inherit from
the corresponding ROOT histogram class depending on the value of the ``type``
keyword argument supplied to the constructor. The ROOT base class is always the
last class in the inheritance list. For example, to create a 2-dimensional
histogram with floating-point bin contents::

   >>> from rootpy.plotting import Hist2D
   >>> hist = Hist2D(10, 0, 1, 5, 0, 1, type='F')
   >>> hist.__class__.__bases__
   (<class 'rootpy.plotting.hist._Hist2D'>, <class 'ROOT.TH2F'>)

``type`` can be any of the type corresponding to the ROOT histogram classes:
``"C"``, ``"S"``, ``"I"``, ``"F"``, ``"D"`` (or lower case)::
   
   >>> hist = Hist2D(10, 0, 1, 5, 0, 1, type='C')
   >>> hist.__class__.__bases__
   (<class 'rootpy.plotting.hist._Hist2D'>, <class 'ROOT.TH2C'>)
   >>> hist = Hist2D(10, 0, 1, 5, 0, 1, type='D')
   >>> hist.__class__.__bases__
   (<class 'rootpy.plotting.hist._Hist2D'>, <class 'ROOT.TH2D'>)

If ``type`` is not specified the default is to use floating-point bins
(``"F"``).

Notice that the name or title of the histograms were not specified; In rootpy
names and titles are optional. ROOT will complain if you give two objects the
same name::

   >>> import ROOT
   >>> h1 = ROOT.TH1D("a name", "a title", 10, 0, 1)
   >>> h2 = ROOT.TH1D("a name", "a different title", 10, 0, 1)
   TROOT::Append:0: RuntimeWarning: Replacing existing TH1: a name (Potential memory leak).

With rootpy, if the ``name`` keyword argument is not specified in the
constructor, a universally unique identifier (UUID) will be used instead::

   >>> from rootpy.plotting import Hist
   >>> h = Hist(10, 0, 1)
   >>> h.GetName()
   '089163932f454725a9ab108437db9773'
   >>> h.GetTitle()
   ''

This insures that all objects have unique names by default in rootpy and is
particularly helpful if naming isn't important in your application (since you
don't intend to write the objects in a file, for example). You can, however,
specify a ``name`` and/or ``title``::

   >>> from rootpy.plotting import Hist
   >>> h = Hist(10, 0, 1, name="some name", title="some title")
   >>> h.GetName()
   'some name'
   >>> h.GetTitle()
   'some title'


Plotting Style
==============

