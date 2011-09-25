#!/usr/bin/env python

from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.types import *
from random import gauss, randint
import ROOT

f = open("test.root", "recreate")

class Event(TreeModel):
    
    a_x = Float()
    a_y = Float()
    a_z = Float()
    
    b_x = Float()
    b_y = Float()
    b_z = Float()

    col_x = ROOT.vector("float")()
    col_y = ROOT.vector("float")()
    col_z = ROOT.vector("float")()
    col_n = Int()

    i = Int()

tree = Tree("test", model=Event)

for i in xrange(10000):
    tree.a_x = gauss(.5, 1.)
    tree.a_y = gauss(.3, 2.)
    tree.a_z = gauss(13., 42.)
    
    tree.b_x = gauss(.5, 1.)
    tree.b_y = gauss(.3, 2.)
    tree.b_z = gauss(13., 42.)
    
    n = randint(1, 10)
    for j in xrange(n):
        tree.col_x.push_back(gauss(.5, 1.))
        tree.col_y.push_back(gauss(.3, 2.))
        tree.col_z.push_back(gauss(13., 42.))
    tree.col_n = n

    tree.i = i
    tree.fill(reset=True)
tree.write()

tree.define_object(name='a', prefix='a_')
tree.define_object(name='b', prefix='b_')
tree.define_collection(name='particles', prefix='col_', size='col_n')

for event in tree:
    print event.a.x
    print event.b.y
    for p in event.particles:
        print p.x

f.close()
