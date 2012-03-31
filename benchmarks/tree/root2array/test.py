#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open

from rootpy.root2array import tree_to_recarray, \
                              tree_to_recarray_c

import cProfile
import time

with open('test.root') as f:

    tree = f.test
    branches = ["a_x", "a_y", "a_z"]

    print "Using pure Python method..."
    cProfile.run('arr1 = tree_to_recarray(tree, branches=branches)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr1 = tree_to_recarray(tree, branches=branches)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40

    print "Using compiled C extension..."
    cProfile.run('arr2 = tree_to_recarray_c(tree, branches=branches)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr2 = tree_to_recarray_c(tree, branches=branches)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40
    print "Comparison of output:"

    print arr1
    print arr2

    print arr1['a_x']
    print arr2['a_x']
