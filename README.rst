.. -*- mode: rst -*-

About
=====

rootpy aims to provide a more feature-rich and pythonic interface with the `ROOT <http://root.cern.ch/>`_ libraries
on top of the `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ interface.

Install
=======

python setup.py install --user

For improved performance set the shell environment variable $ROOTPY_DATA
to a permanent directory (i.e. ~/.rootpy). By default all temporary data (compiled ROOT dictionaries)
are stored in a temporary directory and deleted upon exit.

Examples
========

see examples/*
