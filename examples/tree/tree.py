#!/usr/bin/env python

from rootpy.tree import Tree
from roopty.io import open
import random

f = open("test.root", "recreate")

tree = Tree("test")
tree.branches([('x', 'F'),
               ('y', 'F'),
               ('z', 'F'),
               ('i', 'I')])

for i in xrange(1000):
    tree.x = random.gaussian(.5, 1.)
    tree.y = random.gaussian(.3, 2.)
    tree.z = random.gaussian(13., 42.)
    tree.fill()
tree.write()

f.close()
