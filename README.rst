.. -*- mode: rst -*-

.. image:: https://travis-ci.org/rootpy/rootpy.png
   :target: https://travis-ci.org/rootpy/rootpy

rootpy: Pythonic ROOT
=====================

   `rootpy` provides a more feature-rich pythonic interface
   with the `ROOT <http://root.cern.ch/>`_ libraries on top of
   the existing `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ bindings.

Python has become the language of choice for high-level applications where
fast prototyping and efficient development are important, while
glueing together low-level libraries for performance-critical tasks.
The `PyROOT`_ bindings introduced ROOT into the Python arena, however,
interacting with ROOT in Python should not "feel" like you are writing C++.
Python also offers a multitude of powerful packages such as
`SciPy <http://www.scipy.org/>`_,
`NumPy <http://numpy.scipy.org/>`_,
`IPython <http://ipython.org/>`_,
`matplotlib <http://matplotlib.sourceforge.net/>`_, 
and `PyTables <http://www.pytables.org/>`_,
but a suitable interface between them and ROOT has been lacking.

The rootpy project is a community-driven initiative aiming to provide a more
pythonic interface with ROOT on top of the existing PyROOT bindings.
Several key features include:

* Improvements to help you create and manipulate trees, histograms, cuts
  and vectors.

* Colours and other style attributes can be referred to by descriptive strings.

* ``Get`` and ``Set`` methods are also properties.

* You can define objects and object collections whose properties are tree
  branches.

* Easy navigation through ROOT files. You can now access objects with
  ``my_file.some_directory.tree_name``, for example.

* Dictionaries for STL types such as ``std::vector`` (arbitrarily nested)
  are compiled for you automatically.

* The ability to redirect ROOT error messages through Python's logging system,
  optionally turning them into Python exceptions. 

* Plot your ROOT histograms or graphs with `matplotlib`_.

* Conversion of ROOT trees into `NumPy`_ `ndarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
  and `recarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_.
  Now take advantage of the many statistical and numerical packages
  that Python offers (`NumPy`_, `SciPy`_,
  `StatsModels <http://statsmodels.sourceforge.net/>`_,
  and `scikit-learn <http://scikit-learn.org>`_).

* Conversion of ROOT files containing trees into
  `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format with
  `PyTables`_.

* ``roosh``, a Bash-like shell environment for the ROOT file

* ``rootpy``, a command for common tasks such as summing histograms or drawing
  tree expressions over multiple files, listing the contents of a file,
  or inspecting tree branches and their sizes and types.


Documentation
-------------

Documentation is hosted here:
`rootpy.org <http://rootpy.org>`_
and mirrored here:
`rootpy.github.com/rootpy <http://rootpy.github.com/rootpy>`_.


Requirements
------------

* Python 2.6 or 2.7 (Python 3 is currently not supported, but see
  `this issue <https://github.com/rootpy/rootpy/issues/35>`_ for progress)

* `ROOT`_ 5.28+ with `PyROOT`_ enabled

The following dependencies are optional:

* `NumPy`_ and `root_numpy <https://github.com/rootpy/root_numpy>`_ for speed
* `matplotlib`_ for plotting
* `PyTables`_ for HDF5 IO in rootpy.root2hdf5
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
