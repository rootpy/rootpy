"""
will implement matplotlib.pyplot-like module here
"""

from .plotting import *

def figure(*args, **kwargs):

    return Canvas(*args, **kwargs)
