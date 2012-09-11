"""
attach_event_handler(canvas, handler=close_on_esc_or_middlemouse)
    Attach a handler function to the ProcessedEvent slot, defaulting to
    closing when middle mouse is clicked or escape is pressed

    Note that escape only works if the pad has focus, which in ROOT-land means
    the mouse has to be over the canvas area.
"""

import os
from pkg_resources import resource_filename

import ROOT as R

def load_macro(cpp_code):
    """
    Attempt to load a C++ macro relative to the directory of this python file
    """
    filename = resource_filename('rootpy', 'etc/%s' % cpp_code)
    basename, filename = os.path.split(filename)
    if basename not in R.gROOT.GetMacroPath().split(':'):
        R.gROOT.SetMacroPath('%s:%s' % (basename, R.gROOT.GetMacroPath()))
    R.gROOT.LoadMacro(filename)

def get_process_events_dispatcher():
    """
    Load the event handler macro if it isn't already loaded.
    """
    if hasattr(R, "TPyDispatcherProcessedEvent"):
        return R.TPyDispatcherProcessedEvent
    load_macro("pydispatcher_processed_event.cpp+")
    return R.TPyDispatcherProcessedEvent

def close_on_esc_or_middlemouse(event, x, y, obj):
    """
    Closes canvases when escape is pressed or the canvas area is clicked with
    the middle mouse button.
    (ROOT requires that the mouse is over the canvas area
     itself before sending signals of any kind)
    """
    #print "Event handler called:", args


    if (event == R.kButton2Down
            # User pressed middle mouse
        or (event == R.kMouseMotion and x == y == 0 and R.gROOT.IsEscaped())
            # User pressed escape. Yes. Don't ask me why kMouseMotion.
        ):

        # Defer the close because otherwise root segfaults when it tries to
        # run gPad->Modified()
        obj._py_closetimer = R.TTimer()
        obj._py_closetimer.Connect("Timeout()", "TCanvas", obj, "Close()")
        obj._py_closetimer.Start(10, R.kTRUE) # Single shot after 10ms

def attach_event_handler(canvas, handler=close_on_esc_or_middlemouse):
    """
    Attach a handler function to the ProcessedEvent slot
    """

    if getattr(canvas, "_py_event_dispatcher_attached", None):
        return

    event_dispatcher = get_process_events_dispatcher()(handler)
    canvas.Connect("ProcessedEvent(int,int,int,TObject*)",
                   "TPyDispatcherProcessedEvent", event_dispatcher,
                   "Dispatch(int,int,int,TObject*)")

    # Attach a handler only once to each canvas, and keep the dispatcher alive
    canvas._py_event_dispatcher_attached = event_dispatcher
