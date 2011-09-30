#!/usr/bin/env python

from rootpy.io import open, DoesNotExist

f = open('data.root')

print f.a
print f.a.b

try:
    print f.a.b.c.d.e.f
except DoesNotExist, e:
    print e

for thing in f.walk():
    print thing

f.close()

# supports with statements
with open('data.root') as f:
    print f
