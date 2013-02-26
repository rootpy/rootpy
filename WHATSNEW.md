What's new in `rootpy`
----------------------

A lot of discussion happens in github's issues and pull requests. We
track which pull requests are included in a version of rootpy (see 
[here](https://github.com/rootpy/rootpy/issues/139) for an example) so if you
"watch" a pull request which fixes an issue you care about you can be
automatically notified when it is fixed in a public release.

You might also like to search our
[issues](https://github.com/rootpy/rootpy/issues).

# 0.7

Tracked in [#139](https://github.com/rootpy/rootpy/issues/139)

## bugfixes

    * Fix #49. Don't require python>=2.6 for install, otherwise we get python3
    * Check for sys.version < (2, 6) in setup.py
    * Eliminate ipython printing (Bool_t)1 when ROOT starts up
    * Emit an error wherever the diaper pattern is used
    * Sanitize ROOT's message output to ascii, using repr() if it isn't
    * Register HistStack with class registry correctly

## general improvements

    * Documentation
    * Defer configuring defaults until finalSetup (#74)
    * Switch to pyparsing for C++ template parsing
    * Use XDG base directory specificiation
    * Add defaults.extra_initializations for further ROOT setup
    * Add userdata.BINARY_PATH, unique per (arch, rootversion)
    * Use .cache/rootpy/{arch}-{rootversion}/{modules,dicts} to store generated
      binaries
    * defaults: Disable AutoDict generation
    * Introduce rootpy/util/quickroot which can get to vital root symbols
      without finalSetup (10x speed improvement in import time)
    * Implement TPyDispatcherProcessedEvent using rootpy.compiled
    * Falling back to batch mode should be a WARNING not ERROR
    * Numpy code is now moved to the rootpy/root_numpy repository
    * Use hash of unique name rather than generating a UUID for dictionary
      shared object name
    * Automatic testing of each commit / pull request on travis-ci
    * PEP8 Python code style compliance
    * Add TitleView class to plotting.views functionality       
    * Unit tests
    * Rewrite registry code so magical imports are no longer required to make
      rootpy aware of what ROOT classes are subclassed in rootpy
    * Alias ROOT CamelCase methods with snake_case methods
    * Improvements to the setup.py script
    * New examples for using Trees, Hists, etc
    * Make name and title properties for Objects

## new externs

    * Add extern/pyparsing.py
    * Add extern/byteplay
    * Add module_facade
    * Add inject_closure_values to close over global variables

## docs

    * Add logger documentation
    * Sphinx: Introduce ipython highlighting
    * Document a bit more about python's vanilla logging
    * Update STL docs
    * Add note in CONTRIBUTING.rst about debug python builds

## scripts

    * Use argparse (argparse is now in rootpy.extern)
    * Unify scripts into one rootpy script
    * Remove scripts that are still WIP
    * Use the new rootpy logger
    * Your roosh command history is saved and can be searched
    * Multi-canvas and multi-file support in roosh
    * Improvements to the root2hdf5 script, that now uses root_numpy for faster
      conversion to HDF5

## rootpy.interactive (NEW!)

    * Add interactive.rootwait, provides wait_for_zero_canvases which blocks
      program execution until all canvases are closed
    * Add canvas_events for closing canvases on middle click

## rootpy.context (NEW!)

    * Add preserve_{current_canvas,batch_state} and invisible_canvas context
      managers
    * Add preserve_current_directory context manager

## rootpy.plotting

    * Add xaxis/yaxis properties to plotting.Efficiency
    * Implement new fill_array function in root_numpy to fill histograms with 
      NumPy arrays. Histograms in rootpy now have a fill_array method
    * TStyle is now subclasses in rootpy as Style. Styles can be used at context
      managers using the "with" statement
    * Add the official ATLAS style and example
    * Hist2Ds can be "ravel()"ed, like NumPy's ravel() method that converts a 2D
      array into a 1D array by repeating the second axis along the first
    * Add SetColor to all Plottables to set all colors simultaneously
    * Improvements to the root2matplotlib module
    * Make style attributes properties
    * Add xaxis, yaxis, zaxis properties for Hist classes
    * Plottable now gracefully handles deprecation of properties

## rootpy.logger (NEW!)

    * New logging module for internal and (optionally) external use
    * Automatically coloured status level if we're attached to a terminal
    * Default rootpy logging level to INFO unless os.environ['DEBUG'] is present
    * Redirecting ROOT's errors as python exceptions    
    * Add a `log` symbol to each subpackage    
    * Add @log.trace() decorator which can be used to log function 
      entry/exit/duration
    * Add log.showstack() to log current stack whenever a message is emitted by
      `log`
    * If there is no logging handler when the first message is emitted,
      automatically add one
    * Automatically log python stack trace if there is a segfault    

## rootpy.compiled (NEW!)

    * Adds an interface for compiling C++ code on demand with CompileMacro
    * Add support for inline C++ code definitions
    * rootpy.compiled.register_file("mycode.cxx", ["mysymbol"]) then 
      rootpy.compiled.mysymbol will be generated when requested.
      mycode.cxx is located relative to the module where register_file is called
    * rootpy.compiled.register_code(".. C++ source code ..", ["mysymbol"])
      .. same as above but without needing a cxx file on disk

## rootpy.memory (NEW!)

    * Add log.showdeletion() to show TObject cleanup
    * keepalive: function to ensure objects are kept alive exactly as long as
      needed

## rootpy.util.hook (NEW!)

    * Add ability to hook ROOT methods and append properties/methods to existing
      classes

## rootpy.util.cpp (NEW!)

    * New module for parsing C++ syntax.

## rootpy.stl (NEW!)

    * Compiling arbitrary templated types.

## rootpy.tree

    * Make root_numpy an optional dependency and remove copy from rootpy
    * Reading branches on demand: Only call GetEntry on branches that are
      accessed
    * Add support for branches of array types
    * Improvements to the documentation
    * The categories module is rewritten and now properly parses the category
      syntax (unit tests are included)
    * Improvements to Tree.Draw
    * Dictionaries for branches of STL types are now compiled automatically
    * Reverse order of axes expressions in Tree.Draw to match hist argument
      order so X:Y:Z maps onto filling along X:Y:Z instead of Z:Y:X
