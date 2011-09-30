#!/usr/bin/env python

from rootpy.tree import Tree
from rootpy.io import open
from time import time

f = open("test.root")

tree = f.test


for cached in (True, False):
    
    tree.use_cache(cached)

    start_time = time()
    for event in tree:
        event.x
    end_time = time()
    print "%.2fsec to read one branch" % (end_time - start_time)

    start_time = time()
    for event in tree:
        event.x
        event.y
    end_time = time()
    print "%.2fsec to read two branches" % (end_time - start_time)

    start_time = time()
    for event in tree:
        event.x
        event.y
        event.z
    end_time = time()
    print "%.2fsec to read three branches" % (end_time - start_time)

f.close()
