#!/usr/bin/env python

# this import is required to register the Tree class
import rootpy.tree
from rootpy.io import root_open
from time import time
from ROOT import TTreeCache
import sys

for cached in (False, True):

    try:
        f = root_open("test.root")
    except IOError:
        sys.exit("test.root does not exist. Please run tree_write.py first.")
    tree = f.test

    if cached:
        TTreeCache.SetLearnEntries(1)
        tree.SetCacheSize(10000000)
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

    print "Reading %i bytes in %i transactions" % (f.GetBytesRead(), f.GetReadCalls())
    f.close()
