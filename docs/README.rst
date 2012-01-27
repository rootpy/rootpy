.. -*- mode: rst -*-

Building the docs
=================

First initialize and pull the submodule containing the Sphinx theme::

    git submodule init
    git submodule update

To build the docs as html
(you need `Sphinx <http://sphinx.pocoo.org/>`_ installed)::

    make html

or simply (html is the default)::

    make

The built html is then found in `_build/html`


Uploading the docs
==================

To upload the docs to `http://{user}.github.com/rootpy/` install
`ghp-import <http://pypi.python.org/pypi/ghp-import>`_::

    pip install ghp-import

and from the top directory of `rootpy` (not here in `docs`)::

    ghp-import -p docs/_build/html/
