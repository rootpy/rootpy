# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from .. import compiled as C

__all__ = [
    'close_on_esc_or_middlemouse',
    'attach_event_handler',
]

C.register_code("""
#include <TPython.h>
#include <TPyDispatcher.h>

class TPyDispatcherProcessedEvent : public TPyDispatcher {
public:
    TPyDispatcherProcessedEvent(PyObject* callable) : TPyDispatcher(callable){}

    PyObject* Dispatch(int p1, int p2, int p3, void* p4) {
        if (!p4) return NULL;
        PyObject* p4_aspyobj = TPython::ObjectProxy_FromVoidPtr(p4,
            reinterpret_cast<TObject*>(p4)->ClassName());
        PyObject* result = DispatchVA("lllO", p1, p2, p3, p4_aspyobj);
        return result;
    }

    ClassDef(TPyDispatcherProcessedEvent, 0);
};

ClassImp(TPyDispatcherProcessedEvent);
""", ["TPyDispatcherProcessedEvent"])


def close_on_esc_or_middlemouse(event, x, y, obj):
    """
    Closes canvases when escape is pressed or the canvas area is clicked with
    the middle mouse button. (ROOT requires that the mouse is over the canvas
    area itself before sending signals of any kind.)
    """
    #print "Event handler called:", args

    if (event == ROOT.kButton2Down
            # User pressed middle mouse
        or (event == ROOT.kMouseMotion and
            x == y == 0 and
            # User pressed escape. Yes. Don't ask me why kMouseMotion.
            ROOT.gROOT.IsEscaped())):

        # Defer the close because otherwise root segfaults when it tries to
        # run gPad->Modified()
        obj._py_closetimer = ROOT.TTimer()
        obj._py_closetimer.Connect("Timeout()", "TCanvas", obj, "Close()")
        # Single shot after 10ms
        obj._py_closetimer.Start(10, ROOT.kTRUE)


def attach_event_handler(canvas, handler=close_on_esc_or_middlemouse):
    """
    Attach a handler function to the ProcessedEvent slot, defaulting to
    closing when middle mouse is clicked or escape is pressed

    Note that escape only works if the pad has focus, which in ROOT-land means
    the mouse has to be over the canvas area.
    """
    if getattr(canvas, "_py_event_dispatcher_attached", None):
        return

    event_dispatcher = C.TPyDispatcherProcessedEvent(handler)
    canvas.Connect("ProcessedEvent(int,int,int,TObject*)",
                   "TPyDispatcherProcessedEvent", event_dispatcher,
                   "Dispatch(int,int,int,TObject*)")

    # Attach a handler only once to each canvas, and keep the dispatcher alive
    canvas._py_event_dispatcher_attached = event_dispatcher
