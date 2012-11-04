#!/usr/bin/env python
"""
This benchmark compares the fill_array Hist method with Python's map() and
NumPy's histogram function.
"""
print __doc__
from rootpy.plotting import Hist
import numpy as np
import cProfile
import time

h = Hist(1000, -5, 5)
array = np.random.randn(1E6)

print "Using Python's map()..."
cProfile.run('map(h.Fill, array)')

h.Reset()

print "time without profiler overhead:"
t1 = time.time()
map(h.Fill, array)
t2 = time.time()
print "%f seconds" % (t2 - t1)

h.Reset()

print
print '=' * 40
print

print "Using NumPy's histogram..."
cProfile.run('np.histogram(array)')

h.Reset()

print "time without profiler overhead:"
t1 = time.time()
np.histogram(array)
t2 = time.time()
print "%f seconds" % (t2 - t1)

h.Reset()

print
print '=' * 40
print

print "Using compiled C extension..."
cProfile.run('h.fill_array(array)')

h.Reset()

print "time without profiler overhead:"
t1 = time.time()
h.fill_array(array)
t2 = time.time()
print "%f seconds" % (t2 - t1)
