#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open

from rootpy.root2array import tree_to_recarray, \
                              tree_to_recarray_c

import cProfile
import time

with open('test.root') as f:

    tree = f.test

    print "Using pure Python method..."
    cProfile.run('arr1 = tree_to_recarray(tree)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr1 = tree_to_recarray(tree)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40

    print "Using compiled C extension..."
    cProfile.run('arr2 = tree_to_recarray_c(tree)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr1 = tree_to_recarray_c(tree)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40
    print "Comparison of output:"

    print arr1
    print arr2

    print arr1['a_x']
    print arr2['a_x']
