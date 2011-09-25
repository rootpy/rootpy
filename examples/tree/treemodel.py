#!/usr/bin/env python

from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.types import *
from random import gauss

f = open("test.root", "recreate")

class Event(TreeModel):
    a_x = Float()
    a_y = Float()
    a_z = Float()
    b_x = Float()
    b_y = Float()
    b_z = Float()
    i = Int()

tree = Tree("test", model=Event)

for i in xrange(10000):
    tree.a_x = gauss(.5, 1.)
    tree.a_y = gauss(.3, 2.)
    tree.a_z = gauss(13., 42.)
    tree.b_x = gauss(.5, 1.)
    tree.b_y = gauss(.3, 2.)
    tree.b_z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

tree.define_object(name='a', prefix='a_')
tree.define_object(name='b', prefix='b_')

for event in tree:
    print event.a.x
    print event.b.y

f.close()
