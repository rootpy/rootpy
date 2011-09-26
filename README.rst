.. -*- mode: rst -*-

About
=====

rootpy aims to provide a more feature-rich and pythonic interface
with the `ROOT <http://root.cern.ch/>`_ libraries on top of
the `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ interface.

More specifically, rootpy provides:

* an interface between ROOT and `matplotlib <http://matplotlib.sourceforge.net/>`_.
  Don't like the way your plots look in ROOT? Simply use rootpy to
  plot your ROOT histograms or graphs with matplotlib instead

* easier manipulation of histograms and graphs. rootpy wraps ROOT histograms and graphs
  and implements the arithmetic operators

* an easy way to create and read ROOT TTrees and a mechanism for defining
  objects and collections of objects whose attributes are TTree branches

* easy navigation through TFiles. rootpy wraps TFile and implements the
  natural naming convention so that objects may be retrieved with
  myFile.someDirectory.treeName, for example

* the ability to convert ROOT TFiles containing TTrees into
  `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format with `PyTables <http://www.pytables.org/>`_

* a framework for parallelizing processes which run over many TTrees

* a collection of useful command line scripts

Install
=======

The easiest way to install rootpy is via ``easy_install``.
To install for all users::

    sudo easy_install rootpy

To install in your home directory::

    easy_install --user rootpy

If you have obtained a copy of rootpy yourself use the ``setup.py``
script to install. To install for all users::

    sudo python setup.py install

To install in your home directory::

    python setup.py install --user

For improved performance set the shell environment variable $ROOTPY_DATA
to a permanent directory (i.e. ~/.rootpy). By default all temporary data (compiled ROOT dictionaries)
are stored in a temporary directory and deleted upon exit.

Examples
========

see examples/*
