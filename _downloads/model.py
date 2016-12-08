#!/usr/bin/env python
"""
==================================
Tree models and object collections
==================================

This example demonstrates how to define a tree model and collections of objects
associated to sets of tree branches.
"""
print(__doc__)
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
from rootpy.vector import LorentzVector
from rootpy import stl
from random import gauss, randint


f = root_open("test.root", "recreate")

# define the model
class Point(TreeModel):
    x = FloatCol()
    y = FloatCol()
    z = FloatCol()

class Event(Point.prefix('a_'), Point.prefix('b_')):
    # a_x, a_y, a_z and b_x, b_y, b_z are implicitly included here
    # define vector branches
    col_x = stl.vector("float")
    col_y = stl.vector("float")
    col_z = stl.vector("float")
    col_n = IntCol()
    # a TLorentzVector
    p = LorentzVector
    i = IntCol()

tree = Tree("test", model=Event)

# fill the tree
for i in range(10):
    tree.a_x = gauss(.5, 1.)
    tree.a_y = gauss(.3, 2.)
    tree.a_z = gauss(13., 42.)

    tree.b_x = gauss(.5, 1.)
    tree.b_y = gauss(.3, 2.)
    tree.b_z = gauss(13., 42.)

    n = randint(1, 10)
    for j in range(n):
        tree.col_x.push_back(gauss(.5, 1.))
        tree.col_y.push_back(gauss(.3, 2.))
        tree.col_z.push_back(gauss(13., 42.))
    tree.col_n = n

    tree.p.SetPtEtaPhiM(gauss(.5, 1.),
                        gauss(.5, 1.),
                        gauss(.5, 1.),
                        gauss(.5, 1.))

    tree.i = i
    tree.fill(reset=True)
tree.write()

f.close()
f = root_open("test.root")

tree = f.test

# define objects by prefix:
tree.define_object(name='a', prefix='a_')
tree.define_object(name='b', prefix='b_')

# define a mixin class to add functionality to a tree object
class Particle(object):
    def who_is_your_daddy(self):
        print("You are!")

# define collections of objects by prefix
tree.define_collection(name='particles',
                       prefix='col_',
                       size='col_n',
                       mix=Particle)

# loop over "events" in tree
for event in tree:
    print("a.x: {0:f}".format(event.a.x))
    print("b.y: {0:f}".format(event.b.y))
    # loop over "particles" in current event
    for p in event.particles:
        print("p.x: {0:f}".format(p.x))
        p.who_is_your_daddy()
    print(event.p.Eta())

f.close()
