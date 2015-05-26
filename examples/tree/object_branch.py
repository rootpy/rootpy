#!/usr/bin/env python
"""
===============
Object branches
===============

This simple example demonstrates how to define a TreeModel with a branch of type
std::vector<TLorentzVector>.
"""
print(__doc__)
from rootpy.vector import LorentzVector
from rootpy.tree import Tree, TreeModel, IntCol
from rootpy.io import root_open
from rootpy import stl
from random import gauss


f = root_open("test.root", "recreate")

# define the model
class Event(TreeModel):
    x = stl.vector('TLorentzVector')
    i = IntCol()

tree = Tree("test", model=Event)

# fill the tree
for i in range(100):
    tree.x.clear()
    for j in range(5):
        vect = LorentzVector(
            gauss(.5, 1.),
            gauss(.5, 1.),
            gauss(.5, 1.),
            gauss(.5, 1.))
        tree.x.push_back(vect)
    tree.i = i
    tree.fill()

tree.write()
f.close()
