#!/usr/bin/env python
from rootpy.interactive import wait
from rootpy.plotting import Hist
from rootpy.root2array import fill_hist_with_ndarray
import numpy as np

a = Hist(1000, -5, 5)
fill_hist_with_ndarray(a, np.random.randn(1000000))
a.draw()
wait(True)
