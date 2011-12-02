.. -*- mode: rst -*-

About
=====

rootpy aims to provide a more feature-rich and pythonic interface
with the `ROOT <http://root.cern.ch/>`_ libraries on top of
the existing `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ interface.

More specifically, rootpy provides:

* an interface between ROOT and `matplotlib <http://matplotlib.sourceforge.net/>`_.
  Don't like the way your plots look in ROOT? Simply use rootpy to
  plot your ROOT histograms or graphs with matplotlib instead

* easier manipulation of histograms, graphs, and TVector/TLorentzVectors.
  rootpy provides classes that inherit from these ROOT classes
  and implement the Python arithmetic operators

* an easy way to create and read ROOT TTrees and a mechanism for defining
  objects and collections of objects whose attributes are TTree branches.
  You may also decorate TTree objects with additional methods and attributes.
  See examples/tree.

* easy navigation through TFiles. rootpy wraps TFile and implements the
  natural naming convention so that objects may be retrieved with
  myFile.someDirectory.treeName, for example

* the ability to convert ROOT TFiles containing TTrees into
  `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format with `PyTables <http://www.pytables.org/>`_

* a framework for parallelizing processes that run over many TTrees

* ``roosh``, a Bash-like shell environment for the ROOT TFile

* a collection of useful command line scripts: ``root-ls``, ``root-cp``, ``root-tree-ls``, and others.


Requirements
============

At least Python version 2.6 and
`ROOT <http://root.cern.ch/>`_ with `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ enabled.
`matplotlib <http://matplotlib.sourceforge.net/>`_, `numpy <http://numpy.scipy.org/>`_,
`PyTables <http://www.pytables.org/>`_, and `PyYAML <http://pyyaml.org/>`_ are optional.


Install
=======

The easiest way to install rootpy is with ``pip``.
To install for all users::

    sudo pip install rootpy

To install in your home directory::

    pip install --user rootpy

If you have obtained a copy of rootpy yourself use the ``setup.py``
script to install. To install for all users::

    sudo python setup.py install

To install in your home directory::

    python setup.py install --user

To install the optional requirements
(first download a source distribution if you haven't already)::

    pip install -U -r optional-requirements.txt


Getting the latest source
=========================

Clone the code from github.com with git::

    git clone git://github.com/ndawe/rootpy.git

or checkout with svn::

    svn checkout http://svn.github.com/ndawe/rootpy


Examples
========

see examples/*


Developers Wanted
=================

Please contact me (Noel dot Dawe AT cern dot ch) if you have ideas or contributions.
And of course feel free to fork rootpy at GitHub.com and later submit a pull request.

Currently, rootpy needs attention in these areas:

* Documentation
* Tutorials
* A website displaying the above (currently here: `http://ndawe.github.com/rootpy <http://ndawe.github.com/rootpy>`_)
* Unit testing
* Brenchmarking performance (i.e. Tree read/write)
* Finishing the server/worker code for distributed computing across multiple nodes
* Creation of a TBrowser alternative using PyGTK
* Creation of a framework for managing datasets (using SQLite as a back-end? with revision control?)
