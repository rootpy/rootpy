.. include:: references.txt

.. raw:: html

   <h1 class="main">rootpy: Pythonic ROOT</h1>

.. include:: ../README.rst
   :start-line: 13
   :end-line: 78

Simple Integration into Existing Applications
---------------------------------------------

The `rootpy.ROOT` module is intended to be a drop-in replacement for
ordinary PyROOT imports by mimicking PyROOT's interface. Both ROOT and rootpy
classes can be accessed in a harmonized way through this module. This means you
can take advantage of rootpy classes automatically by replacing ``import ROOT``
with ``import rootpy.ROOT as ROOT`` or ``from rootpy import ROOT`` in your code,
while maintaining backward compatibility with existing use of ROOT's classes.

Under `rootpy.ROOT`, classes are automatically "asrootpy'd" *after* the ROOT
constructor has been called:

.. sourcecode:: python

    >>> import rootpy.ROOT as ROOT
    >>> ROOT.TH1F('name', 'title', 10, 0, 1)
    Hist('name')

Access rootpy classes under `rootpy.ROOT` without needing to remember
where to import them from in rootpy:

.. sourcecode:: python

    >>> import rootpy.ROOT as ROOT
    >>> ROOT.Hist(10, 0, 1, name='name', type='F')
    Hist('name')


User Guide
==========

.. toctree::
   :numbered:
   :maxdepth: 2

   install
   cern
   modules/plotting
   modules/trees
   modules/files
   modules/logger


API Reference
=============

.. toctree::
   :numbered:
   :maxdepth: 1

   modules/classes


Examples
========

.. toctree::
   :maxdepth: 1

   auto_examples/index

Development
===========

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

Post your questions on `stackoverflow.com
<http://stackoverflow.com/questions/tagged/rootpy>`_
and use the tag ``rootpy``.

Think you found a bug? Open a new issue here:
`github.com/rootpy/rootpy/issues <https://github.com/rootpy/rootpy/issues>`_.

Also feel free to post questions or follow discussion on the
`rootpy-users <http://groups.google.com/group/rootpy-users>`_ or
`rootpy-dev <http://groups.google.com/group/rootpy-dev>`_ Google groups.
