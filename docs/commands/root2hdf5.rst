.. root2hdf5

.. include:: ../references.txt

=========
root2hdf5
=========

rootpy provides the command ``root2hdf5`` for converting ROOT files containing
TTrees into HDF5 files containing HDF5 tables.

.. note::
   To use this command you must have HDF5_, PyTables_, NumPy_ and root_numpy_
   installed.

Run with the help option to see a full list of available options:

.. command-output:: root2hdf5 -h

Typical output will look like this:

.. command-output:: root2hdf5 ../rootpy/testdata/test_tree.root

When run interactively you will also see a progress bar fill from left to right
as each tree is converted. The progress bar can be disabled with the
``--no-progress-bar`` option.

.. note::
   Also see rootpy's :py:func:`rootpy.root2hdf5.root2hdf5` function for direct
   access to the underlying conversion function for use in your own
   applications.
