.. -*- mode: rst -*-

.. image:: https://travis-ci.org/rootpy/rootpy.png
   :target: https://travis-ci.org/rootpy/rootpy

rootpy: Pythonic ROOT
=====================

Python has become the language of choice for high-level applications where
fast prototyping and efficient development are important, while
glueing together low-level libraries for performance-critical tasks.
The `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ bindings introduced
`ROOT <http://root.cern.ch/>`_ into the Python arena, however, interacting with
ROOT in Python should not "feel" like you are writing C++. Python also offers a
multitude of powerful packages such as
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

* ``Get`` and ``Set`` methods on ROOT objects are also properties.

* Provides a way of mapping ROOT trees onto python objects and collections.

* Easy navigation through ROOT files. You can now access objects with
  ``my_file.some_directory.tree_name``, for example.

* Dictionaries for STL types are compiled for you automatically.

* Redirect ROOT's messages through Python's logging system.

* Optionally turn ROOT errors into Python exceptions.

* Plot your ROOT histograms or graphs with `matplotlib`_.

* Conversion of ROOT trees into `NumPy`_ `ndarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_
  and `recarrays
  <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
  through the related `root_numpy <https://github.com/rootpy/root_numpy>`_
  package. Now take advantage of the many statistical and numerical packages
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

View the documentation at `rootpy.org <http://rootpy.org>`_
or (a possibly more up-to-date version) at
`rootpy.github.com/rootpy <http://rootpy.github.com/rootpy>`_.

Also see an introductory presentation here:
`rootpy.org/intro.pdf <http://rootpy.org/intro.pdf>`_


Requirements
------------

* Python 2.6 or 2.7 (Python 3 is currently not supported, but see
  `this issue <https://github.com/rootpy/rootpy/issues/35>`_ for progress)

* `ROOT`_ 5.28+ with `PyROOT`_ enabled

The following dependencies are optional:

* `NumPy`_ and `root_numpy`_ for speed
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

Try `rootpy` on `CERN's LXPLUS <http://information-technology.web.cern.ch/services/lxplus-service>`_
----------------------------------------------------------------------------------------------------

First, `set up ROOT <http://root.cern.ch/drupal/content/starting-root>`_::

    source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5/setup.sh &&\
    cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.00/x86_64-slc5-gcc43-opt/root &&\
    source bin/thisroot.sh &&\
    cd -

Then, create and activate a `virtualenv <https://pypi.python.org/pypi/virtualenv>`_ (change `my_env` at your will)::

    virtualenv my_env # necessary only the first time
    source my_env/bin/activate

Get the `latest source <https://github.com/rootpy/rootpy#getting-the-latest-source>`_::

    git clone https://github.com/rootpy/rootpy.git

and `install <https://github.com/rootpy/rootpy#manual-installation>`_ it::

    ~/my_env/bin/python rootpy/setup.py install

Note that neither `sudo` nor `--user` is used, because we are in a virtualenv.

`rootpy` should now be ready to `use <https://github.com/rootpy/rootpy#documentation>`_::

    python
    >>> import rootpy

Post-Installation
-----------------

If you installed `rootpy` into your home directory with the `--user` option
above, add ``${HOME}/.local/bin`` to your ``${PATH}`` if it is not there
already (put this in your .bashrc)::

   export PATH=${HOME}/.local/bin${PATH:+:$PATH}

Running the Tests
-----------------

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once rootpy is installed, it may be tested (from outside the source directory)
by running::

   nosetests --exe -v -a '!slow' rootpy

rootpy can also be tested before installing by running this from inside the
source directory::

   make test


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
