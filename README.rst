.. -*- mode: rst -*-

`[see full documentation] <http://rootpy.org>`_

rootpy: Pythonic ROOT
=====================

.. image:: https://img.shields.io/pypi/v/rootpy.svg
   :target: https://pypi.python.org/pypi/rootpy
.. image:: https://travis-ci.org/rootpy/rootpy.png
   :target: https://travis-ci.org/rootpy/rootpy
.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.18897.svg
   :target: http://dx.doi.org/10.5281/zenodo.18897

Python has become the language of choice for high-level applications where
fast prototyping and efficient development are important, while
glueing together low-level libraries for performance-critical tasks.
The `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ bindings introduced
`ROOT <http://root.cern.ch/>`_ into the world of Python, however, interacting
with ROOT in Python should not feel like you are still writing C++.

The rootpy project is a community-driven initiative aiming to provide a more
pythonic interface with ROOT on top of the existing PyROOT bindings. Given
Python's reflective and dynamic nature, rootpy also aims to improve ROOT design
flaws and supplement existing ROOT functionality. The scientific Python
community also offers a multitude of powerful packages such as
`SciPy <http://www.scipy.org/>`_,
`NumPy <http://numpy.scipy.org/>`_,
`matplotlib <http://matplotlib.sourceforge.net/>`_,
`scikit-learn <http://scikit-learn.org>`_,
and `PyTables <http://www.pytables.org/>`_,
but a suitable interface between them and ROOT has been lacking. rootpy
provides the interfaces and conversion mechanisms required to liberate your
data and to take advantage of these alternatives if needed.

Key features include:

* Improvements to help you create and manipulate trees, histograms, cuts
  and vectors.

* Dictionaries for STL types are compiled for you automatically.

* Redirect ROOT's messages through Python's logging system.

* Optionally turn ROOT errors into Python exceptions.

* ``Get`` and ``Set`` methods on ROOT objects are also properties.

* Easy navigation through ROOT files. You can now access objects with
  ``my_file.some_directory.tree_name``, for example.

* Colours and other style attributes can be referred to by descriptive strings.

* Provides a way of mapping ROOT trees onto python objects and collections.

* Plot your ROOT histograms or graphs with `matplotlib`_.

* Conversion of ROOT trees into `NumPy`_ `ndarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
  and `recarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
  through the related `root_numpy <http://rootpy.github.io/root_numpy/>`_
  package. Now take advantage of the many statistical and numerical packages
  that Python offers (`NumPy`_, `SciPy`_,
  `StatsModels <http://statsmodels.sourceforge.net/>`_,
  and `scikit-learn`_).

* Conversion of ROOT files containing trees into
  `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format with
  `PyTables`_.

* ``roosh``, a Bash-like shell environment for the ROOT file, very useful for
  quick ROOT file inspection and interactive plotting.

* ``rootpy``, a command for common tasks such as summing histograms or drawing
  tree expressions over multiple files, listing the contents of a file,
  or inspecting tree branches and their sizes and types.

