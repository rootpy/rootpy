# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module supports monitoring TObject deletions.

.. warning::
   This is not recommended for production

"""
from __future__ import absolute_import

from weakref import ref
import ctypes
from ctypes import CFUNCTYPE, py_object, addressof, c_int

from .. import compiled as C
from .. import QROOT, log
from ..utils.cinterface import callback, objectproxy_realaddress


__all__ = [
    'monitor_deletion',
    'monitor_object_deletion',
]


def monitor_deletion():
    """
    Function for checking for correct deletion of weakref-able objects.

    Example usage::

        monitor, is_alive = monitor_deletion()
        obj = set()
        monitor(obj, "obj")
        assert is_alive("obj") # True because there is a ref to `obj` is_alive
        del obj
        assert not is_alive("obj") # True because there `obj` is deleted

    """
    monitors = {}

    def set_deleted(x):
        def _(weakref):
            del monitors[x]
        return _

    def monitor(item, name):
        monitors[name] = ref(item, set_deleted(name))

    def is_alive(name):
        return monitors.get(name, None) is not None

    return monitor, is_alive


cleanuplog = log["memory.cleanup"]
cleanuplog.show_stack()

# Add python to the include path
C.add_python_includepath()

C.register_code("""
    #ifndef __CINT__
    #include <Python.h>
    #endif
    #include <TObject.h>
    #include <TPython.h>

    class RootpyObjectCleanup : public TObject {
    public:
        typedef void (*CleanupCallback)(PyObject*);
        CleanupCallback _callback;

        RootpyObjectCleanup(CleanupCallback callback) : _callback(callback) {}

        virtual void RecursiveRemove(TObject* object) {
            // When arriving here, object->ClassName() will _always_ be TObject
            // since we're called by ~TObject, and virtual method calls don't
            // work as expected from there.
            PyObject* o = TPython::ObjectProxy_FromVoidPtr(object, "TObject");

            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);

            _callback(o);

            PyErr_Restore(ptype, pvalue, ptraceback);
            PyGILState_Release(gstate);
        }

        ClassDef(RootpyObjectCleanup, 0);
    };

    ClassImp(RootpyObjectCleanup);

""", ["RootpyObjectCleanup"])

MONITORED = {}


@CFUNCTYPE(None, py_object)
def on_cleanup(tobject):
    # Note, when we arrive here, tobject is in its ~TObject, and hence the
    # subclass part of the object doesn't exist, in some sense. Hence why we
    # store information about the object on the MONITORED dict.
    addr = objectproxy_realaddress(tobject)
    if addr in MONITORED:
        args = MONITORED[addr]
        fn, args = args[0], args[1:]
        fn(tobject, *args)
        del MONITORED[addr]

initialized = False


def init():
    global initialized
    if initialized: return
    initialized = True

    cleanup = C.RootpyObjectCleanup(callback(on_cleanup))

    cleanups = QROOT.gROOT.GetListOfCleanups()
    cleanups.Add(cleanup)

    import atexit

    @atexit.register
    def exit():
        # Needed to ensure we don't get called after ROOT has gone away
        cleanups.RecursiveRemove(cleanup)


def monitor_object_deletion(o, fn=lambda *args: None):

    init()

    # Required so that GetListOfCleanups().RecursiveRemove() is called.
    o.SetBit(o.kMustCleanup)

    args = fn, type(o).__name__, o.GetName(), o.GetTitle(), repr(o)
    MONITORED[objectproxy_realaddress(o)] = args
