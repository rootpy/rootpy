This is the class and function reference of rootpy.

.. contents:: List of modules
   :local:

.. _compiled_ref:

:mod:`rootpy.compiled`: Compiling C++
=====================================

.. automodule:: rootpy.compiled
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compiled.register_code
   compiled.register_file

.. _context_ref:

:mod:`rootpy.context`: Context Managers
=======================================

.. automodule:: rootpy.context
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: function.rst

   context.preserve_current_style
   context.preserve_current_canvas
   context.preserve_current_directory
   context.preserve_batch_state
   context.invisible_canvas

.. _cross_validation_ref:

:mod:`rootpy.data`: Dataset Management
======================================

.. automodule:: rootpy.data
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   data.dataset.Fileset
   data.dataset.Treeset

.. _interactive_ref:

:mod:`rootpy.interactive`: Interactive Plotting
===============================================

.. automodule:: rootpy.interactive
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: function.rst

   interactive.wait_for_zero_canvases
   interactive.wait

.. _io_ref:

:mod:`rootpy.io`: ROOT I/O
==========================

.. automodule:: rootpy.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   io.Directory
   io.File
   io.TemporaryFile

.. autosummary::
   :toctree: generated/
   :template: function.rst

   io.open
   io.walk
   io.rm
   io.cp
   io.mkdir

.. _logger_ref:

:mod:`rootpy.logger`: Logging
=============================

.. automodule:: rootpy.logger
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   logger.LogFilter

.. autosummary::
   :toctree: generated/
   :template: function.rst

   logger.log_trace

.. _memory_ref:

:mod:`rootpy.memory`: Memory Management
=======================================

.. automodule:: rootpy.memory
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. _plotting_ref:

:mod:`rootpy.plotting`: Plotting
================================

.. automodule:: rootpy.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plotting.Hist
   plotting.Hist2D
   plotting.Hist3D
   plotting.HistStack
   plotting.Graph
   plotting.Graph2D
   plotting.Legend
   plotting.Canvas
   plotting.Pad

.. _root2hdf5_ref:

:mod:`rootpy.root2hdf5`: Conversion to HDF5
===========================================

.. automodule:: rootpy.root2hdf5
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: function.rst

   root2hdf5.convert

.. _stl_ref:

:mod:`rootpy.stl`: STL Dictionary Generation
============================================

.. automodule:: rootpy.stl
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated
   :template: function.rst

   stl.generate

.. _tree_ref:

:mod:`rootpy.tree`: Trees
=========================

.. automodule:: rootpy.tree
   :no-members:
   :no-inherited-members:

.. currentmodule:: rootpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.Tree
   tree.TreeModel
   tree.TreeChain
   tree.TreeQueue
   tree.Cut
   tree.Categories
   tree.ObjectCol
   tree.BoolCol
   tree.BoolArrayCol
   tree.CharCol
   tree.CharArrayCol
   tree.UCharCol
   tree.UCharArrayCol
   tree.ShortCol
   tree.ShortArrayCol
   tree.UShortCol
   tree.UShortArrayCol
   tree.IntCol
   tree.IntArrayCol
   tree.UIntCol
   tree.UIntArrayCol
   tree.LongCol
   tree.LongArrayCol
   tree.ULongCol
   tree.ULongArrayCol
   tree.FloatCol
   tree.FloatArrayCol
   tree.DoubleCol
   tree.DoubleArrayCol
