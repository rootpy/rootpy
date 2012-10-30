"""
This module handles creation of the user-data area
"""
import os
import tempfile
import atexit


DATA_ROOT = None
if os.getenv('ROOTPY_GRIDMODE') not in ('1', 'true'):
    DATA_ROOT = os.getenv('ROOTPY_DATA')
    if DATA_ROOT is None:
        DATA_ROOT = os.path.expanduser('~/.rootpy')
    else:
        DATA_ROOT = os.path.expandvars(os.path.expanduser(DATA_ROOT))
    # check if expanduser failed:
    if DATA_ROOT.startswith('~'):
        DATA_ROOT = None
    elif not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    elif not os.path.isdir(DATA_ROOT):
        # A file at DATA_ROOT already exists
        DATA_ROOT = None

__is_tmp = False
if DATA_ROOT is None:
    print "Warning: placing user data in /tmp.\n" \
          "Make sure ~/.rootpy or $ROOTPY_DATA\n" \
          "is a writable directory so that I don't need to\n" \
          "recreate all user data each time"
    DATA_ROOT = tempfile.mkdtemp()
    __is_tmp = True


@atexit.register
def __cleanup():
    if __is_tmp:
        import shutil
        shutil.rmtree(DATA_ROOT)
