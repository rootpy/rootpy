.. -*- mode: rst -*-

rootpy
======

Documentation is hosted here: `rootpy.org <http://rootpy.org>`_
and mirrored here:
`rootpy.github.com/rootpy <http://rootpy.github.com/rootpy>`_.

About
-----

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

Requirements
------------

rootpy is developed and tested on Linux and Mac.

* Python 2.6 or 2.7
  At the moment python 3 is not supported, but see `this issue <https://github.com/rootpy/rootpy/issues/35>`_ for progress.
* `ROOT`_ 2.28+ with `PyROOT`_ enabled

Optional
--------

* `NumPy`_ for speed
* `matplotlib`_ for plotting
* `PyTables`_ for HDF5 IO in rootpy.root2tables
* `readline <http://docs.python.org/library/readline.html>`_ and `termcolor <http://pypi.python.org/pypi/termcolor>`_ for roosh

..
   NumPy: which min version? List all places required in rootpy.
   matplotlib: which min version? List all places required in rootpy.


Getting the Latest Source
-------------------------

Clone the repository with git::

    git clone git://github.com/rootpy/rootpy.git
    
or checkout with svn::

    svn checkout http://svn.github.com/rootpy/rootpy

Note: svn does not checkout git submodules so you will end up with an
incomplete `rootpy`.

Still using svn? Watch `this <http://www.youtube.com/watch?v=4XpnKHJAok8>`_.


Manual Installation
-------------------

If you have obtained a copy of `rootpy` yourself use the ``setup.py``
script to install.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

    python setup.py install --user

To install system-wide (requires root privileges)::

    sudo python setup.py install

To install optional requirements (`matplotlib`_, `NumPy`_, etc.)::

    pip install -U -r optional-requirements.txt

To install roosh requirements::

    pip install -U -r roosh-requirements.txt

To disable building the extension modules, do this before installing::

    export ROOTPY_NO_EXT=1


Automatic Installation
----------------------

The easiest way to install a `released version
<http://pypi.python.org/pypi/rootpy/>`_ of
`rootpy` is with `pip <http://pypi.python.org/pypi/pip>`_.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

    pip install --user rootpy

To install system-wide (requires root privileges)::

    sudo pip install rootpy


Post-Installation
-----------------

If you installed `rootpy` into your home directory with the `--user` option
above, add ``${HOME}/.local/bin`` to your ``${PATH}`` if it is not there
already (put this in your .bashrc)::

   export PATH=${HOME}/.local/bin${PATH:+:$PATH}


Examples
--------

see examples/*


Developers Wanted
-----------------

Please post on the rootpy-dev@googlegroups.com list if you have ideas
or contributions. Feel free to fork
`rootpy on GitHub <https://github.com/rootpy/rootpy>`_
and later submit a pull request.

`rootpy` needs attention in these areas:

* Documentation
* Tutorials
* A website displaying the above (currently under construction here:
  `http://rootpy.github.com/rootpy <http://rootpy.github.com/rootpy>`_)
* Unit testing
* Brenchmarking performance (i.e. Tree read/write)
* Cleaning up the existing code
* Finishing the server/worker code for distributed computing across multiple
  nodes
* Creation of a TBrowser alternative using PyGTK
* Creation of a framework for managing datasets (using SQLite as a back-end?
  with revision control?)
* Additional features anyone would like to implement


Have Questions or Found a Bug?
------------------------------

Post your questions on `stackoverflow.com <http://stackoverflow.com/>`_
and use the tag ``rootpy`` (this tag does not exist yet, but if you have a
reputation of at least 1500 then please create it).

Think you found a bug? Open a new issue here:
`github.com/rootpy/rootpy/issues <https://github.com/rootpy/rootpy/issues>`_.

Also feel free to post questions or follow discussion on the developers
(rootpy-dev@googlegroups.com) or users (rootpy-users@googlegroups.com) lists.
