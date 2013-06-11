#!/usr/bin/env python
"""
=================================
Convert a tree into a NumPy array
=================================

This example demonstrates how to convert a Tree into a NumPy ndarray or
recarray.
"""
print __doc__
import rootpy
rootpy.log.basic_config_colorized()
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
from random import gauss


f = root_open("test.root", "recreate")


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
from root_numpy import tree2rec
array = tree2rec(tree)
print array
print array.x
print array.i
print tree.to_array()

f.close()
