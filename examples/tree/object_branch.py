#!/usr/bin/env python
"""
===============
Object branches
===============

This simple example demonstrates how to define a TreeModel with a branch of type
std::vector<TLorentzVector>.
"""
print __doc__
import ROOT
from rootpy.math.physics.vector import LorentzVector
from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.classfactory import generate
from rootpy.types import IntCol
from random import gauss

# this is already done for you in rootpy.types
# writing it here again as an example
# since this dictionary already exists, it won't be generated again
generate('vector<TLorentzVector>', 'TLorentzVector.h')

f = open("test.root", "recreate")


# define the model
class Event(TreeModel):

    x = ROOT.vector('TLorentzVector')
    i = IntCol()

tree = Tree("test", model=Event)

# fill the tree
for i in xrange(100):
    tree.x.clear()
    for j in xrange(5):
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
