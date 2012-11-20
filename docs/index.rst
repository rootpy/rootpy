.. raw:: html
   
   <h1>rootpy</h1>

`rootpy` provides a more feature-rich and pythonic interface
with the `ROOT <http://root.cern.ch/>`_ libraries on top of
the existing `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ bindings.

More specifically, `rootpy` provides:

* easier manipulation of trees, histograms, graphs, cuts,
  and TVector/TLorentzVectors. `rootpy` provides classes that
  inherit from these ROOT classes and implement the Python
  arithmetic operators.

* an easy way to create and read ROOT TTrees and a mechanism for defining
  objects and collections of objects whose attributes are TTree branches.
  You may also decorate TTree objects with additional methods and attributes.
  See examples/tree.

* easy navigation through TFiles. `rootpy` wraps TFile and implements the
  natural naming convention so that objects may be retrieved with
  myFile.someDirectory.treeName, for example.

* an interface between ROOT and
  `matplotlib <http://matplotlib.sourceforge.net/>`_.
  Don't like the way your plots look in ROOT? Simply use `rootpy` to
  plot your ROOT histograms or graphs with matplotlib instead.

* conversion of ROOT TFiles containing TTrees into
  `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format with
  `PyTables <http://www.pytables.org/>`_.

* conversion of TTrees into `NumPy <http://numpy.scipy.org/>`_ `ndarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
  and `recarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_.
  Now take advantage of the many statistical and numerical packages
  that Python offers (`NumPy`_, `SciPy <http://www.scipy.org/>`_,
  `StatsModels <http://statsmodels.sourceforge.net/>`_,
  use `scikit-learn <http://scikit-learn.org>`_ for machine
  learning instead of `TMVA <http://tmva.sourceforge.net/>`_).

* efficient filling of ROOT histograms from `NumPy` `ndarrays`.

* a framework for parallelizing processes that run over many TTrees.

* ``roosh``, a Bash-like shell environment for the ROOT TFile.

* a collection of useful command line scripts: ``root-ls``, ``root-cp``,
  ``root-tree-ls``, and others.


User Guide
==========

.. toctree::
   :numbered:
   :maxdepth: 1

   user_guide.rst


Examples                                                                  
========
                                                                                 
.. toctree::                                                                     
   :maxdepth: 2                                                                  
                                                                                 
   auto_examples/index
