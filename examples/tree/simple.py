#!/usr/bin/env python
"""
=====================
A simple Tree example
=====================

This example demonstrates how to create a simple tree.
"""
print(__doc__)
from rootpy.tree import Tree
from rootpy.io import root_open
from random import gauss

f = root_open("test.root", "recreate")

tree = Tree("test")
tree.create_branches(
    {'x': 'F',
     'y': 'F',
     'z': 'F',
     'i': 'I'})

for i in range(100):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

f.close()
