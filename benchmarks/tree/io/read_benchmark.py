#!/usr/bin/env python
"""
==========================
Comparing read performance
==========================

This example demonstrates differences in read performance with and without
reading branches on demand.
"""
print __doc__
from rootpy.tree import Tree
from rootpy.io import root_open
import random
import string
import time

def random_name(length):
    return ''.join(random.choice(string.ascii_letters) for x in xrange(length))

f = root_open("test.root", "recreate")


def create_tree(num_branches, num_entries):

    branches = dict([(random_name(10), 'F') for x in xrange(num_branches)])
    tree = Tree()
    tree.create_branches(branches)
    # fill the tree with zeros
    for i in xrange(num_entries):
        tree.fill()
    return tree

for on_demand in (False, True):
    for num_branched_read in xrange(1, 50, 10):
        tree = create_tree(1000, 10000)
        names = tree.branchnames
        tree.read_branches_on_demand = on_demand
        tree.SetCacheSize(10000000)
        tree.SetCacheLearnEntries(10)
        read_names = random.sample(names, num_branched_read)
        t0 = time.time()
        # loop on the tree
        for event in tree:
            # getattr the branches
            for name in read_names:
                getattr(tree, name)
            pass
        t1 = time.time()
        print on_demand, num_branched_read, t1 - t0
f.close()
