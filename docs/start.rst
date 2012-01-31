===============
Getting Started
===============

Requirements
------------

At least Python version 2.6 and
`ROOT <http://root.cern.ch/>`_ with `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ enabled.
`matplotlib <http://matplotlib.sourceforge.net/>`_, `numpy <http://numpy.scipy.org/>`_,
`PyTables <http://www.pytables.org/>`_, and `PyYAML <http://pyyaml.org/>`_ are optional.


Install
-------

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
-------------------------

Clone the code from `github <http://github.com>`_ with git::

    git clone git://github.com/rootpy/rootpy.git

or checkout with svn::

    svn checkout http://svn.github.com/rootpy/rootpy

