#!/usr/bin/env python
"""
=======================
A simple Ntuple example
=======================

This example demonstrates how to create a simple Ntuple.
"""
print(__doc__)
from rootpy.tree import Ntuple
from rootpy.io import root_open
from random import gauss

f = root_open("test.root", "recreate")

# create an ntuple with three float fields: a, b, c
ntuple = Ntuple(('a', 'b', 'c'), name="test")

# fill the ntuple with random data
for i in range(20):
    ntuple.Fill(gauss(.5, 1.), gauss(.3, 2.), gauss(13., 42.))
ntuple.write()

# write as CSV
ntuple.csv()

f.close()
