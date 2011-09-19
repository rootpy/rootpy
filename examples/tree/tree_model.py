#!/usr/bin/env python

from rootpy.tree import Tree, TreeModel
from rootpy.io import open
from rootpy.types import *
import random

f = open("test.root", "recreate")

class Particle(TreeModel):
    x = Float()
    y = Float()
    z = Float()
    i = Int()

tree = Tree("test", model=Particle)

for i in xrange(10000):
    tree.x = random.gauss(.5, 1.)
    tree.y = random.gauss(.3, 2.)
    tree.z = random.gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

f.close()
