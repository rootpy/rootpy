#!/usr/bin/env python

# this import is required
import rootpy.tree
from rootpy.io import root_open

from rootpy.root2array import tree_to_recarray_py, \
                              tree_to_recarray, \
                              tree_to_ndarray

import cProfile
import time

with root_open('test.root') as f:

    tree = f.test
    branches = ["a_x", "a_y", "a_z"]
    tree.SetWeight(.5)

    print "Using pure Python method..."
    cProfile.run('arr1 = tree_to_recarray_py(tree, branches=branches,'
                 'include_weight=True)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr1 = tree_to_recarray_py(tree, branches=branches,
                               include_weight=True)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40

    print "Using compiled C extension..."
    cProfile.run('arr2 = tree_to_recarray(tree, branches=branches,'
                 'include_weight=True)')

    print "time without profiler overhead:"
    t1 = time.time()
    arr2 = tree_to_recarray(tree, branches=branches,
                            include_weight=True)
    t2 = time.time()
    print "%f seconds" % (t2 - t1)

    print '=' * 40
    print "Comparison of output:"

    print
    print "Python result:"
    print arr1
    print arr1['a_x']
    print arr1['weight']

    print
    print "C result:"
    print arr2
    print arr2['a_x']
    print arr2['weight']

    arr3 = tree_to_ndarray(tree, branches=branches,
                           include_weight=True)

    print
    print "C result as ndarray:"
    print arr3
