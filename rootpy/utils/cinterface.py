# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Functions useful for interfacing with C/C++ functions:

* ``callback`` => Allows you to pass ctypes CFUNCTYPE objects as parameters to
              PyROOT functions
* ``objectproxy_realaddress`` => Determine the real address of a ROOT objects
    (useful because multiple ObjectProxies can point to the same underlying object)
"""
from __future__ import absolute_import

import ctypes as C

from . import quickroot as QROOT

__all__ = [
    'callback',
    'objectproxy_realaddress',
]


def callback(cfunc):
    """
    Turn a ctypes CFUNCTYPE instance into a value which can be passed into PyROOT
    """
    # Note:
    # ROOT wants a c_voidp whose addressof() == the call site of the target
    # function. This hackery is necessary to achieve that.
    return C.c_voidp.from_address(C.cast(cfunc, C.c_voidp).value)


def objectproxy_realaddress(obj):
    """
    Obtain a real address as an integer from an objectproxy.
    """
    voidp = QROOT.TPython.ObjectProxy_AsVoidPtr(obj)
    return C.addressof(C.c_char.from_buffer(voidp))
