.. -*- mode: rst -*-

.. image:: https://travis-ci.org/rootpy/rootpy.png
   :target: https://travis-ci.org/rootpy/rootpy

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


Requirements
------------

* Python 2.6 or 2.7 (Python 3 is currently not supported, but see
  `this issue <https://github.com/rootpy/rootpy/issues/35>`_ for progress)

* `ROOT`_ 5.28+ with `PyROOT`_ enabled

The following dependencies are optional:

* `NumPy`_ for speed
* `matplotlib`_ for plotting
* `PyTables`_ for HDF5 IO in rootpy.root2tables
* `readline <http://docs.python.org/library/readline.html>`_ and
  `termcolor <http://pypi.python.org/pypi/termcolor>`_ for roosh

rootpy is developed and tested on Linux and Mac.

..
   NumPy: which min version? List all places required in rootpy.
   matplotlib: which min version? List all places required in rootpy.


Getting the Latest Source
-------------------------

Clone the repository with git::

    git clone git://github.com/rootpy/rootpy.git
    
or checkout with svn::

    svn checkout http://svn.github.com/rootpy/rootpy

.. note:: svn does not checkout git submodules so you will end up with an
   incomplete `rootpy`.


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

    pip install -U -r requirements/[roosh|array|...].txt


Automatic Installation
----------------------

To install a `released version
<http://pypi.python.org/pypi/rootpy/>`_ of
`rootpy` use `pip <http://pypi.python.org/pypi/pip>`_.

.. note:: This will install the latest version of rootpy on PyPI which may be
   lacking many new unreleased features.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

    pip install --user rootpy

To install system-wide (requires root privileges)::

    sudo pip install rootpy

To install optional requirements (`matplotlib`_, `NumPy`_, etc.)::

    pip install --user rootpy[array,matplotlib,...]

This requires
`pip version 1.1 <http://www.pip-installer.org/en/latest/news.html#id3>`_ 
or later.


Post-Installation
-----------------

If you installed `rootpy` into your home directory with the `--user` option
above, add ``${HOME}/.local/bin`` to your ``${PATH}`` if it is not there
already (put this in your .bashrc)::

   export PATH=${HOME}/.local/bin${PATH:+:$PATH}


Development
-----------

Please post on the rootpy-dev@googlegroups.com list if you have ideas
or contributions. Feel free to fork
`rootpy on GitHub <https://github.com/rootpy/rootpy>`_
and later submit a pull request.


IRC Channel
-----------

See #rootpy on freenode.

IRC is banned at CERN since it reveals your hostname to people in the chatroom,
making you interesting to attackers. But you can safely access it through this
web link:

http://webchat.freenode.net/?randomnick=1&channels=rootpy&prompt=1


Have Questions or Found a Bug?
------------------------------

Post your questions on `stackoverflow.com <http://stackoverflow.com/>`_
and use the tag ``rootpy`` (this tag does not exist yet, but if you have a
reputation of at least 1500 then please create it).

Think you found a bug? Open a new issue here:
`github.com/rootpy/rootpy/issues <https://github.com/rootpy/rootpy/issues>`_.

Also feel free to post questions or follow discussion on the
`rootpy-users <http://groups.google.com/group/rootpy-users>`_ or
`rootpy-dev <http://groups.google.com/group/rootpy-dev>`_ Google groups.
