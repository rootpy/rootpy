#!/usr/bin/env python
"""
=============================================
The TreeChain class, a replacement for TChain
=============================================

This example demonstrates how to use the TreeChain class, a more Python-friendly
TChain replacement.
"""
print(__doc__)
from random import gauss
from rootpy.io import root_open
from rootpy.tree import Tree, TreeChain
from rootpy.plotting import Hist
from rootpy.extern.six.moves import range

# Make two files, each with a Tree called "test"

print("Creating test tree in chaintest1.root")
f = root_open("chaintest1.root", "recreate")

tree = Tree("test")
branches = {
    'x': 'F',
    'y': 'F',
    'z': 'F',
    'i': 'I'}
tree.create_branches(branches)

for i in range(10000):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()

# Make a histogram of x when y > 1
hist1 = Hist(100, -10, 10, name='hist1')
tree.Draw('x', 'y > 1', hist=hist1)
hist1.SetDirectory(0) # memory resident
print("The first tree has {0:f} entries where y > 1".format(hist1.Integral()))

tree.write()
f.close()

print("Creating test tree in chaintest2.root")
f = root_open("chaintest2.root", "recreate")

tree = Tree("test")
tree.create_branches(branches)

for i in range(10000):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

# Make a histogram of the second tree
hist2 = Hist(100, -10, 10, name='hist2')
tree.Draw('x', 'y > 1', hist=hist2)
hist2.SetDirectory(0) # memory resident
print("The second tree has {0:f} entries where y > 1".format(hist2.Integral()))
f.close()

combined_hist = hist1 + hist2

print("Building TreeChain")
chain = TreeChain('test', ['chaintest2.root', 'chaintest1.root'])
# Make the equivalent of the combined_hist
combined_hist_chain = Hist(100, -10, 10, name='combined')
chain.Draw('x', 'y > 1', hist=combined_hist_chain)

residual = combined_hist_chain - combined_hist
print(
    "The combined histogram (separately) minus "
    "the combined from the chain has {0:f} entries".format(
        residual.Integral()))
