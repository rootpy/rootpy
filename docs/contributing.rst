.. _contributing

============
Contributing
============

Please post on the rootpy-dev@googlegroups.com list if you have ideas
or contributions. Feel free to fork
`rootpy on GitHub <https://github.com/rootpy/rootpy>`_
and later submit a pull request.


Running the Tests
=================

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once rootpy is installed, it may be tested (from outside the source directory)
by running::

   nosetests --exe -v -a '!slow' rootpy

rootpy can also be tested before installing by running this from inside the
source directory::

   make test


Writing Documentation
=====================

All classes, methods, functions and modules should be documented according to
the `NumPy/SciPy documentation standard
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Building the docs
-----------------

To build the docs as html
(you need `Sphinx <http://sphinx.pocoo.org/>`_ installed)::

    make

The html is then found in `_build/html`.
To preview the docs locally before uploading::

    make show


Uploading the docs
------------------

If you have not done so already, add a reference to the main rootpy repository
from your fork::

    git remote add upstream https://github.com/rootpy/rootpy.git

Create a local branch that tracks the main ``gh-pages`` branch::

    git fetch upstream
    git branch gh-pages upstream/gh-pages

To upload the docs to `http://rootpy.github.com/rootpy/`::

    make gh-pages

You will be prompted for your username and password.


IRC Channel
===========

See #rootpy on freenode.

IRC is banned at CERN since it reveals your hostname to people in the chatroom,
making you interesting to attackers. But you can safely access it through this
web link:

http://webchat.freenode.net/?randomnick=1&channels=rootpy&prompt=1


Git Usage: Merging and Rebasing
===============================

Read this article for an in-depth discussion on
`git best practices <http://lwn.net/Articles/328436/>`_.

Try to keep your history as simple as possible. Avoid merges on private code
unless there is a particularly good reason to. Instead of merging in
``rootpy/master`` to update your local branch, use rebase instead. Merging in
``rootpy/master`` risks creating criss-cross merges which means you can actually
lose code if you're not careful. Git's merging algorithm is actually quite dumb,
so it's best to keep it simple.

See rootpy's network for a graphical view of rootpy's entire history::

   https://github.com/rootpy/rootpy/network

Let's all try our best to keep this graph as clean as possible.


Developer Notes
===============

Using a debug python build
--------------------------

The following CPython configure arguments can be used to obtain a debug build::

    python-6.6.6/build $ ../configure --enable-shared --with-pydebug --without-pymalloc --prefix=install/path

But beware! You will need to build ROOT against this python build. And beware
that GDB might be linked against python. If that's the case, then it will segfault when it starts
if it picks up an incompatible build::

    $ gdb
    gdb: Symbol `_Py_ZeroStruct' has different size in shared object, consider re-linking
    gdb: Symbol `PyBool_Type' has different size in shared object, consider re-linking
    gdb: Symbol `_Py_NotImplementedStruct' has different size in shared object, consider re-linking
    gdb: Symbol `PyFloat_Type' has different size in shared object, consider re-linking
    gdb: Symbol `_Py_TrueStruct' has different size in shared object, consider re-linking
    gdb: Symbol `_Py_NoneStruct' has different size in shared object, consider re-linking
    Segmentation fault

The way around this is to preload the correct library by setting LD_PRELOAD, and
then unsetting it before your program is executed. For example, this will debug
`my-program-to-debug`::

    LD_PRELOAD=/usr/lib/libpython2.7.so gdb -ex 'set environ LD_PRELOAD' --args my-program-to-debug

Note that you need to set LD_PRELOAD to the right version of python that gdb was
compiled against, which you can find with `ldd $(which gdb)` from a fresh
environment.
