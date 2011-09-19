#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open
from random import gauss

f = open("test.root", "recreate")

tree = Tree("test")
tree.create_branches([('x', 'F'),
                      ('y', 'F'),
                      ('z', 'F'),
                      ('i', 'I')])

for i in xrange(10000):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

f.close()
