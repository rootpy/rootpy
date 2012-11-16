.. -*- mode: rst -*-

Git Usage: Merging and Rebasing
===============================

Read this article for an in-depth discussion on
`git best practices <http://lwn.net/Articles/328436/>`_.

Try to keep your history as simple as possible. Avoid merges on private code 
unless there is a particularly good reason to. Instead of merging in ``rootpy/master``
to update your local branch, use rebase instead. Merging in ``rootpy/master`` risks
creating criss-cross merges which means you can actually lose code if you're
not careful. Git's merging algorithm is actually quite dumb, so it's best to
keep it simple. 

See rootpy's network for a graphical view of rootpy's entire history::

https://github.com/rootpy/rootpy/network

Let's all try our best to keep this graph as clean as possible.


Coding Guidlines
================

TODO


Writing Documentation
=====================

TODO

See docs/README


Submodules
==========

To initialize submodules and update to the currently referenced commit::

    git submodule init
    git submodule update

