.. _files:

=====
Files
=====

.. currentmodule:: rootpy.io

Opening ROOT files with ``rootpy.io.root_open``
===============================================

rootpy provides the :func:`rootpy.io.root_open` function that internally uses
ROOT's ``TFile::Open``, but instead returns a :class:`rootpy.io.File` object.
For example:

.. sourcecode:: python

   >>> from rootpy.io import root_open
   >>> myfile = root_open('some_file.root', 'recreate')
   >>> myfile
   File('some_file.root')
   >>> myfile.__class__.__bases__
   (<class 'rootpy.io.file._DirectoryBase'>, <class 'ROOT.TFile'>)

Additionally, any objects retrieved from a rootpy ``File`` are automatically
cast to the subclass in rootpy, if one exists:

.. sourcecode:: python

   >>> from rootpy.testdata import get_file
   >>> testfile = get_file()
   >>> myhist = testfile.dimensions.hist2d
   >>> myhist
   Hist2D('hist2d')

rootpy's ``File`` class inherits from ROOT's TFile but can additionally act as a
context manager:

.. sourcecode:: python

   from rootpy.io import root_open

   with root_open('some_file.root') as myfile:
       # the file is open in this context
       myhist = myfile.somedirectory.histname.Clone()
       myhist.SetDirectory(0)
   # when the context is left the file is closed

Also, as demonstrated in the example above, contents of files and directories
can be accesses as attributes (``myfile.somedirectory.histname``).


Utilities
=========

rootpy files can be "walked" in a similar way to Python's ``os.walk()``:

.. testcode::

   from rootpy.testdata import get_file

   # use the test file shipped with rootpy
   with get_file() as f:
       # access objects by name as properties of the current dir
       myhist = f.dimensions.hist2d
       # recursively walk through the file
       for path, dirs, objects in f.walk():
           # do something
           print path, dirs, objects

the output of which is:

.. testoutput::
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE
   
    ['dimensions', 'scales', 'means', 'graphs', 'gaps', 'efficiencies'] []
   dimensions [] ['hist2d', 'hist3d']
   scales [] ['hist1', 'hist3', 'hist2', 'hist4']
   means [] ['hist1', 'hist3', 'hist2', 'hist4']
   graphs [] ['tgrapherrors', 'tgraph2d', 'tgraphasymmerrors', 'tgraph']
   gaps [] ['hist1', 'hist3', 'hist2', 'hist4']
   efficiencies [] ['hist1', 'hist3', 'hist2', 'hist4', 'eff3v1', 'eff2v1', 'eff4v1']


Also see :mod:`rootpy.io`.

Temporary Files
===============

rootpy provides a :class:`rootpy.io.TemporaryFile` that when closed is
automatically deleted from the filesystem. This may be useful when creating
temporary objects, such as trees copied with a selection that are no longer
needed after the termination of the program.
