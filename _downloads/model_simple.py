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
from string import ascii_letters

f = root_open("test.root", "recreate")

# define the model
class Event(TreeModel):
    s = CharCol()
    string = CharArrayCol(5)
    x = FloatCol()
    y = FloatCol()
    z = FloatCol()
    f = FloatArrayCol(5)
    num_vals = IntCol()
    # variable-length array
    vals = FloatArrayCol(5, length_name='num_vals')

tree = Tree("test", model=Event)

# fill the tree
for i in range(5):
    tree.s = ord(choice(ascii_letters))
    tree.string = (u''.join(sample(ascii_letters, 4))).encode('ascii')
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    for j in range(5):
        tree.f[j] = gauss(-2, 5)
    tree.num_vals = i
    for j in range(i):
        tree.vals[j] = j
    tree.fill()
tree.write()
tree.vals.reset()

# print tree contents in CSV format
tree.csv()

f.close()
