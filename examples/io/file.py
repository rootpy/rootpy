#!/usr/bin/env python

from rootpy.io import open

f = open('data.root')

print f.a
print f.a.b

for thing in f.walk():
    print thing

f.close()
