"""
The functions in this module provide a way of pausing code execution until
canvases are closed. This can be useful when testing code and you don't want to
keep the objects alive outside of your function.

The wait function can be called repeatedly to pause multiple times.

wait_for_zero_canvases()
    Keeps root alive until CTRL-c is pressed or all canvases are closed

wait_for_zero_canvases(middle_mouse_close=True)
    allows canvases to be closed with the middle mouse button (see below)

wait is shorthand for wait_for_zero_canvases
"""

import threading

import ROOT

from .canvas_events import attach_event_handler


if not ROOT.gROOT.IsBatch():
    _processRootEvents = getattr(ROOT.PyGUIThread, "_Thread__target", None)
    _finishSchedule = getattr(ROOT.PyGUIThread, "finishSchedule", None)
else:
    _processRootEvents = None
    _finishSchedule = None


def start_new_gui_thread():
    """
    Attempt to start a new GUI thread, if possible.

    It is only possible to start one if there was one running on module import.
    """
    assert not ROOT.PyGUIThread.isAlive(), "GUI thread already running!"

    assert _processRootEvents, (
        "GUI thread wasn't started when rootwait was imported, "
        "so it can't be restarted")

    ROOT.keeppolling = 1
    ROOT.PyGUIThread = threading.Thread(None, _processRootEvents, None, (ROOT,))

    ROOT.PyGUIThread.finishSchedule = _finishSchedule
    ROOT.PyGUIThread.setDaemon( 1 )
    ROOT.PyGUIThread.start()

def stop_gui_thread():
    """
    Try to stop the GUI thread. If it was running returns True, otherwise False.
    """
    if not ROOT.PyGUIThread.isAlive():
        return False

    ROOT.keeppolling = 0
    ROOT.PyGUIThread.finishSchedule()
    ROOT.PyGUIThread.join()
    return True

@ROOT.TPyDispatcher
def count_canvases():
    """
    Count the number of active canvases and finish gApplication.Run() if there
    are none remaining.

    incpy.ignore
    """
    if not get_visible_canvases():
        ROOT.gSystem.ExitLoop()

@ROOT.TPyDispatcher
def exit_application_loop():
    """
    Signal handler for CTRL-c to cause gApplication.Run() to finish.

    incpy.ignore
    """
    ROOT.gSystem.ExitLoop()

def get_visible_canvases():
    """
    Return a list of active GUI canvases
    (as opposed to invisible Batch canvases)
    """
    return [c for c in ROOT.gROOT.GetListOfCanvases() if not c.IsBatch()]

def run_application_until_done():

    had_gui_thread = stop_gui_thread()

    ROOT.gApplication._threaded = True
    ROOT.gApplication.Run(True)

    if had_gui_thread:
        start_new_gui_thread()

def wait_for_zero_canvases(middle_mouse_close=False):
    """
    Wait for all canvases to be closed, or CTRL-c.

    If `middle_mouse_close`, middle click will shut the canvas.

    incpy.ignore
    """

    # Handle CTRL-c
    sh = ROOT.TSignalHandler(ROOT.kSigInterrupt, True)
    sh.Add()
    sh.Connect("Notified()", "TPyDispatcher", exit_application_loop, "Dispatch()")

    visible_canvases = get_visible_canvases()

    for canvas in visible_canvases:
        if middle_mouse_close:
            attach_event_handler(canvas)

        if not getattr(canvas, "_py_close_dispatcher_attached", False):
            # Attach a handler only once to each canvas
            canvas._py_close_dispatcher_attached = True
            canvas.Connect("Closed()", "TPyDispatcher",
                           count_canvases, "Dispatch()")

    if visible_canvases and not ROOT.gROOT.IsBatch():
        run_application_until_done()

wait = wait_for_zero_canvases

def prevent_close_with_canvases():
    """
    Register a handler which prevents python from exiting until
    all canvases are closed
    """
    register(wait_for_zero_canvases)

def test():
    c = ROOT.TCanvas()
    c.Update()
    wait()

    c2 = ROOT.TCanvas()
    c2.Update()
    wait(True) # This canvas can be killed by middle clicking on it or hitting
               # escape whilst it has focus

if __name__ == "__main__":
    test()
