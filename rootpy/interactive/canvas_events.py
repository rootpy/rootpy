"""
attach_event_handler(canvas, handler=close_on_esc_or_middlemouse)
    Attach a handler function to the ProcessedEvent slot, defaulting to
    closing when middle mouse is clicked or escape is pressed

    Note that escape only works if the pad has focus, which in ROOT-land means
    the mouse has to be over the canvas area.
"""

import os
from pkg_resources import resource_filename

import ROOT

ROOT.gSystem.Load(
        os.path.join(
            os.path.dirname(__file__),
                '_pydispatcher_processed_event.so'))


def close_on_esc_or_middlemouse(event, x, y, obj):
    """
    Closes canvases when escape is pressed or the canvas area is clicked with
    the middle mouse button.
    (ROOT requires that the mouse is over the canvas area
     itself before sending signals of any kind)
    """
    #print "Event handler called:", args


    if (event == ROOT.kButton2Down
            # User pressed middle mouse
        or (event == ROOT.kMouseMotion and x == y == 0 and ROOT.gROOT.IsEscaped())
            # User pressed escape. Yes. Don't ask me why kMouseMotion.
        ):

        # Defer the close because otherwise root segfaults when it tries to
        # run gPad->Modified()
        obj._py_closetimer = ROOT.TTimer()
        obj._py_closetimer.Connect("Timeout()", "TCanvas", obj, "Close()")
        obj._py_closetimer.Start(10, ROOT.kTRUE) # Single shot after 10ms

def attach_event_handler(canvas, handler=close_on_esc_or_middlemouse):
    """
    Attach a handler function to the ProcessedEvent slot
    """

    if getattr(canvas, "_py_event_dispatcher_attached", None):
        return

    event_dispatcher = ROOT.TPyDispatcherProcessedEvent(handler)
    canvas.Connect("ProcessedEvent(int,int,int,TObject*)",
                   "TPyDispatcherProcessedEvent", event_dispatcher,
                   "Dispatch(int,int,int,TObject*)")

    # Attach a handler only once to each canvas, and keep the dispatcher alive
    canvas._py_event_dispatcher_attached = event_dispatcher
