# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
GetOwnership: The analagous function to SetOwnership.

This function is intended for diagnostic purposes and is not guaranteed to keep
working.
"""

import rootpy.compiled as C

C.register_code("""
    #include <sys/types.h>       // for ssize_t

    struct _object;

    struct TFakeObjectProxy {
       ssize_t fRefCnt;          // PyObject_HEAD
       void* fPyType;            // PyObject_HEAD
       void* fRootObj;
       int fFlags;
    };

    bool GetOwnership(_object* obj) {
       return (reinterpret_cast<TFakeObjectProxy*>(obj))->fFlags & 0x0001;
    }
""", ["GetOwnership"])

# Note: this is not a straight assignment because C.GetOwnership causes
#       finalsetup and compilation.

def GetOwnership(obj):
    return C.GetOwnership(obj)
