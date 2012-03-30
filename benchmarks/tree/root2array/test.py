#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open

from rootpy.root2array import tree_to_recarray, \
                              tree_to_recarray_c

with open('test.root') as f:

    tree = f.test

    print tree_to_recarray(tree)
    print tree_to_recarray_c(tree)
