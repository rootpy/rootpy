"""
This module handles creation of the user-data area
"""

import os
import sys
import tempfile
import atexit

DATA_ROOT = os.getenv('ROOTPY_DATA')

__is_tmp = False
if DATA_ROOT is None:
    print "Warning: placing user data in /tmp. " \
          "Set a permanent location with $ROOTPY_DATA for improved performance."
    DATA_ROOT = tempfile.mkdtemp()
    __is_tmp = True
else:
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    if not os.path.isdir(DATA_ROOT):
        sys.exit("A file at %s already exists."% DATA_ROOT)

@atexit.register
def __cleanup():
    if __is_tmp:
        import shutil
        shutil.rmtree(DATA_ROOT)
