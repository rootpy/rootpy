#!/usr/bin/env python

import cProfile
import time
import numpy as np
from rootpy.io import root_open
from rootpy.root2array import tree_to_ndarray
from rootpy.plotting import Hist
# this import is required
import rootpy.tree

h = Hist(10, 0, 1)


def pyroot(tree):

    tree.Draw('a_x', '', 'goff', hist=h)
    v1 = tree.GetV1()
    return set(v1[n] for n in xrange(tree.GetEntries()))


def rootpy(tree):

    return np.unique(tree_to_ndarray(tree, branches=["a_x"]))


with root_open('test.root') as f:

    tree = f.test
    print "Trees has %i entries" % tree.GetEntries()
    print
    print "Using the ROOT/PyROOT way..."
    cProfile.run('pyroot(tree)')
    print "time without profiler overhead:"
    t1 = time.time()
    a = pyroot(tree)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)
    print
    print '=' * 40
    print
    print "Using compiled C extension and numpy..."
    cProfile.run('rootpy(tree)')
    print "time without profiler overhead:"
    t1 = time.time()
    a = rootpy(tree)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)
