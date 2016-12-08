#!/usr/bin/env python
"""
==============================
ROOT.TFile made easy by rootpy
==============================

This example demonstrates how basic file operations are made easier in rootpy.
"""
print(__doc__)
import os
import shutil
from rootpy.io import root_open, DoesNotExist
from rootpy.plotting import Hist, Hist2D
from rootpy import testdata
from rootpy import asrootpy

shutil.copyfile(testdata.get_filepath('test_file_2.root'), 'data.root')
f = root_open('data.root')

print(f.a)
print(f.a.b)

try:
    print(f.a.b.c.d.e.f)
except AttributeError as e:
    print(e)

for thing in f.walk():
    print(thing)

f.close()

# supports with statements
with root_open('data.root', 'update') as f:
    print(f)

    # write some histograms
    h1 = Hist(100, 0, 100, name='h1', type='I')
    # variable bin widths
    h2 = Hist2D((0,3,5,20,50), (10,20,30,40,1000), name='h2')

    h1.Write()
    h2.Write()
# file is automatically closed after with statement

# retrieve the histograms previously saved
with root_open('data.root') as f:

    h1 = f.h1
    # or h1 = f.Get('h1')
    h2 = f.h2
    # or h2 = f.Get('h2')

    # ROOT classes are automatically converted into
    # rootpy form when retrieved from a ROOT file as
    # long as their module was previously imported
    print(h1.__class__.__name__)
    print(h2.__class__.__name__)

    # you may also do this to convert an object into
    # rootpy form (again, assuming the relevant module
    # was previously imported)
    h1 = asrootpy(h1)
    # if it is already in rootpy form or if no rootpy form
    # exists then asrootpy does nothing
    print(h1.__class__.__name__)

os.unlink('data.root')
