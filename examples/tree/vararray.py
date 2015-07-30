#!/usr/bin/env python
"""
=================================
Trees with variable-length arrays
=================================

This example demonstrates how to create a tree with a variable-length array.
"""
print(__doc__)

from rootpy.tree import Tree, TreeModel, IntCol, FloatArrayCol
from rootpy.io import root_open

class Event(TreeModel):
    num_vals = IntCol()
    vals = FloatArrayCol(10, length_name='num_vals')

rfile = root_open('test.root', 'w')
tree = Tree('events', model=Event)

for i in range(10):
    tree.num_vals = i + 1
    for j in range(i + 1):
        tree.vals[j] = j
    tree.fill()

tree.write()
tree.vals.reset()
tree.csv()
rfile.close()
print("===")

# CSV output from tree read from file should match above output
root_open('test.root', 'r').events.csv()
