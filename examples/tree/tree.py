#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open
import random

f = open("test.root", "recreate")

tree = Tree("test")
tree.create_branches([('x', 'F'),
                      ('y', 'F'),
                      ('z', 'F'),
                      ('i', 'I')])

for i in xrange(10000):
    tree.x = random.gauss(.5, 1.)
    tree.y = random.gauss(.3, 2.)
    tree.z = random.gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

f.close()
