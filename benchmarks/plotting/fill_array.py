#!/usr/bin/env python
"""
This benchmark compares the fill_array Hist method with Python's map() and
NumPy's histogram function.
"""
print __doc__
from rootpy.plotting import Hist
import numpy as np
import cProfile
import timeit


h = Hist(1000, -5, 5)
array = np.random.randn(1E6)

def time_repeat(cmd, repeat=5, number=1):
    best_time = min(
            timeit.repeat(cmd,
                repeat=repeat, number=number,
                setup="from __main__ import h, array, np")) / number
    print "%d loops, best of %d: %fs per loop" % (
            number, repeat, best_time)


print "Using Python's map()..."
cProfile.run('map(h.Fill, array)')

h.Reset()

print "time without profiler overhead:"
time_repeat('map(h.Fill, array)')

h.Reset()

print
print '=' * 40
print

print "Using NumPy's histogram..."
cProfile.run('np.histogram(array)')

h.Reset()

print "time without profiler overhead:"
time_repeat('np.histogram(array)')

h.Reset()

print
print '=' * 40
print

print "Using compiled C extension..."
cProfile.run('h.fill_array(array)')

h.Reset()

print "time without profiler overhead:"
time_repeat('h.fill_array(array)')
