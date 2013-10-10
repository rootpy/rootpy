.. _start

===============
Getting Started
===============

Try `rootpy` on `CERN's LXPLUS <http://information-technology.web.cern.ch/services/lxplus-service>`_
====================================================================================================

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

Have Questions or Found a Bug?
==============================

Post your questions on `stackoverflow.com
<http://stackoverflow.com/questions/tagged/rootpy>`_
and use the tag ``rootpy``.

Think you found a bug? Open a new issue here:
`github.com/rootpy/rootpy/issues <https://github.com/rootpy/rootpy/issues>`_.

Also feel free to post questions or follow discussion on the
`rootpy-users <http://groups.google.com/group/rootpy-users>`_ or
`rootpy-dev <http://groups.google.com/group/rootpy-dev>`_ Google groups.
