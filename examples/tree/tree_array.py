#!/usr/bin/env python

from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.types import *
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
from rootpy.root2array import to_numpy_array
array = to_numpy_array(tree)
print array
print array.x
print array.i

f.close()
