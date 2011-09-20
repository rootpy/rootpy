#!/usr/bin/env python

from rootpy.io import open

f = open('data.root')

print f.a
print f.a.b

f.close()
