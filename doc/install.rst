Installation
============

Requirements
------------

Python (at least version 2.6) is required with `ROOT <http://root.cern.ch/drupal/>`_) installed
with the `PyROOT <http://root.cern.ch/drupal/content/pyroot>`_ interface enabled.
The `PyYAML <http://pyyaml.org/wiki/PyYAML>`_ module is required for certain components.
`NumPy <http://numpy.scipy.org/>`_ and `matplotlib <http://matplotlib.sourceforge.net/>`_ are required
if you would like to use that framework instead of ROOT for plotting.

Installing rootpy
-----------------

The `rootpy <http://sourceforge.net/projects/rootpy/>`_ package is also required and can be cloned with git from SourceForge::
    
    git clone git://rootpy.git.sourceforge.net/gitroot/rootpy/rootpy

or checked out from the ATLAS Subversion repository::
    
    svn checkout svn+ssh://${USER}@svn.cern.ch/reps/atlasusr/end/rootpy/trunk rootpy

Install and setup with::

    python setup.py install --prefix=/path/to/install/dir/rootpy
    source /path/to/install/dir/rootpy/etc/rootpy_setup.[c]sh

For your convenience, add the last line above to your .bashrc.
