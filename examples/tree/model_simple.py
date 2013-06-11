#!/usr/bin/env python
"""
===================
A simple tree model
===================

This example demonstrates how to define a simple tree model.
"""
print __doc__
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
for i in xrange(100):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

# write tree in CSV format
tree.csv()

f.close()
