#!/usr/bin/env python
"""
=================================
Convert a tree into a NumPy array
=================================

This example demonstrates how to convert a Tree into a NumPy ndarray or
recarray.
"""
print __doc__
from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.types import FloatCol, IntCol
from random import gauss


f = open("test.root", "recreate")


# define the model
class Event(TreeModel):

    x = FloatCol()
    y = FloatCol()
    z = FloatCol()
    i = IntCol()

tree = Tree("test", model=Event)

# fill the tree
for i in xrange(100000):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

# convert tree into a numpy record array
from rootpy.root2array import tree_to_recarray, tree_to_ndarray
array = tree_to_recarray(tree)
print array
print array.x
print array.i
print tree_to_ndarray(tree)
print tree.recarray()
print tree.ndarray()

f.close()
