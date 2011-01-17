import os
import sys

"""
This module handles creation of the user-data area
(in ~/.rootpy unless changed by the user)
"""

if not os.environ.has_key('ROOTPY_DATA'):
    sys.exit("Shell variable $ROOTPY_DATA is not set! Was setup.[c]sh sourced?")

DATA_ROOT = os.environ['ROOTPY_DATA']

if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)

if not os.path.isdir(DATA_ROOT):
    sys.exit("A file at %s already exists."% DATA_ROOT)
