===============
Getting Started
===============

Watch rootpy do tricks
----------------------

Ever wish browsing through a ROOT file were as easy as navigating a filesystem
in the shell?  Try ``roosh``::

     $ roosh testfile.root
     Welcome to the ROOSH terminal
     type help for help
     testfile.root > ls
     dimensions  efficiencies  gaps  graphs  means  scales
     dimensions > cd means/
     means > ls
     hist1  hist2  hist3  hist4

Ever wish that accessing objects in a ROOT file didn't involve so much writing?
``rootpy`` understands::

    >>> from rootpy.io import File
    >>> from rootpy.test import filename
    >>> testfile = File(filename, 'read')             # File wraps ROOT.TFile
    >>> for top, dirs, objects in testfile: print top # in analogy to os.walk
    ... 
    
    dimensions
    scales
    means
    graphs
    gaps
    efficiencies
    >>> hist = testfile.efficiencies.hist1            # no need for GetDirectory

Ever wish manipulating ROOT objects were more pythonic?  ``rootpy`` does that::

    >>> from rootpy.test import testfile
    >>> hist = testfile.means.hist1
    >>> list(hist)                                    # pythonic access to histogram contents
    [204.0, 478.0, 771.0, 975.0, 947.0, 721.0, 431.0, 238.0, 94.0, 22.0, 6.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> hist[0]                                       # bin indices start at 0
    204.0
    >>> hist[-1]                                      # ...support negative values
    0.0
    >>> hist[3:5]                                     # ...and can handle slicing
    [975.0, 947.0]
    >>> hist.xedges(-1)                               # added convenience methods
    50.0
    >>> hist += testfile.means.hist2                  # operators act like you'd expect
