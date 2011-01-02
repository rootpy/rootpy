import os
import sys

"""
This module handles creation of the user-data area
(in ~/.rootpy unless changed by the user)
"""

if not os.environ.has_key('ROOTPY_DATA'):
    sys.exit("Shell variable $ROOTPY_DATA is not set! Was setup.[c]sh sourced?")

__data_root = os.environ['ROOTPY_DATA']

if not os.path.exists(__data_root):
    os.mkdir(__data_root)

if not os.path.isdir(__data_root):
    sys.exit("A file at %s already exists."% __data_root)
