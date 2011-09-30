#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open

f = open("test.root")

tree = f.test

for event in tree:
    event.x

f.close()
