===============
Getting Started
===============

Ever wish browsing through a ROOT file were as easy as navigating a filesystem
in the shell?  Try ``roosh``::

   $ roosh test_file.root
   Welcome to the ROOSH terminal
   type help for help
   testfile.root > ls
   dimensions  efficiencies  gaps  graphs  means  scales
   dimensions > cd means/
   means > ls
   hist1  hist2  hist3  hist4

Ever wish that accessing objects in a ROOT file didn't involve so much writing?
``rootpy`` understands::

  >>> from rootpy.testdata import get_file
  >>> testfile = get_file('test_file.root')
  >>> for top, dirs, objects in testfile.walk():
  ...     print top # in analogy to os.walk
  <BLANKLINE>
  dimensions
  scales
  means
  graphs
  gaps
  efficiencies
  >>> # no need for GetDirectory
  >>> hist = testfile.efficiencies.hist1

Ever wish manipulating ROOT objects were more pythonic? ``rootpy`` does that::

  >>> from rootpy.testdata import get_file
  >>> testfile = get_file('test_file.root')
  >>> hist = testfile.means.hist1
  >>> # pythonic access to histogram contents
  >>> list(hist)
  [204.0, 478.0, 771.0, 975.0, 947.0, 721.0, 431.0, 238.0, 94.0, 22.0, 6.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  >>> # bin indices start at 0
  >>> hist[0]
  204.0
  >>> # and can handle slicing
  >>> hist[3:5]
  [975.0, 947.0]
  >>> # added convenience methods
  >>> hist.xedges(-1)
  50.0
  >>> # operators act like you'd expect
  >>> hist += testfile.means.hist2
