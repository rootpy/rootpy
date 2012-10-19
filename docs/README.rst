.. -*- mode: rst -*-

Building the docs
=================

First initialize and pull the submodule containing the Sphinx theme.
Do this from the top directory of `rootpy` (not here in `docs`)::

    git submodule init
    git submodule update

To build the docs as html
(you need `Sphinx <http://sphinx.pocoo.org/>`_ installed)::

    make html

or simply (html is the default)::

    make

The built html is then found in `_build/html`.
To preview the docs locally before uploading::

    make show


Uploading the docs
==================

If you have not done so already, add a reference to the main rootpy repository
from your fork::

    git remote add upstream git@github.com:rootpy/rootpy

Create a local branch that tracks the main ``gh-pages`` branch::

    git fetch upstream
    git branch gh-pages upstream/gh-pages

To upload the docs to `http://rootpy.github.com/rootpy/` install
`ghp-import <http://pypi.python.org/pypi/ghp-import>`_::

    pip install ghp-import

and from the top directory of `rootpy` (not here in `docs`)::

    ghp-import -r upstream -p docs/_build/html/

