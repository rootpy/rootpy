#!/usr/bin/env python
"""
===================
A simple tree model
===================

This example demonstrates how to define a simple tree model.
"""
print(__doc__)
from rootpy.tree import Tree, TreeModel
from rootpy.tree import IntCol, FloatCol, FloatArrayCol, CharCol, CharArrayCol
from rootpy.io import root_open
from random import gauss, choice, sample
from string import letters

f = root_open("test.root", "recreate")

# define the model
class Event(TreeModel):
    s = CharCol()
    string = CharArrayCol(5)
    x = FloatCol()
    y = FloatCol()
    z = FloatCol()
    f = FloatArrayCol(5)
    i = IntCol()

tree = Tree("test", model=Event)

# fill the tree
for i in range(100):
    tree.s = choice(letters)
    tree.string = ''.join(sample(letters, 4))
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    for j in range(5):
        tree.f[j] = gauss(-2, 5)
    tree.i = i
    tree.fill()
tree.write()

# write tree in CSV format
tree.csv()

f.close()
