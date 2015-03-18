# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
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

Examples
--------

    from rootpy.plotting import Canvas
    from rootpy.interactive import wait

    c = Canvas()
    c.Update()
    wait()

    c2 = Canvas()
    c2.Update()
    wait(True)
    # This canvas can be killed by middle clicking on it or hitting
    # escape whilst it has focus

"""
from __future__ import absolute_import

import threading
from atexit import register

import ROOT

from . import log; log = log[__name__]
from ..defaults import extra_initialization
from ..memory.keepalive import keepalive
from .. import IN_IPYTHON_NOTEBOOK
from .canvas_events import attach_event_handler

__all__ = [
    'wait_for_zero_canvases',
    'wait_for_browser_close',
    'wait',
]

_processRootEvents = None
_finishSchedule = None
__ACTIVE = False


@extra_initialization
def fetch_vars():
    global _processRootEvents, _finishSchedule, __ACTIVE
    PyGUIThread = getattr(ROOT, 'PyGUIThread', None)
    if PyGUIThread is not None:
        _processRootEvents = getattr(PyGUIThread, "_Thread__target", None)
        _finishSchedule = getattr(PyGUIThread, "finishSchedule", None)
    if _processRootEvents is None:
        if not IN_IPYTHON_NOTEBOOK:
            log.warning(
                """unable to access ROOT's GUI thread either because PyROOT's
                finalSetup() was called while in batch mode or because PyROOT
                is using the new PyOS_InputHook-based mechanism that is not yet
                supported in rootpy (PyConfig.StartGuiThread == 'inputhook' or
                gSystem.InheritsFrom('TMacOSXSystem')). wait() etc. will
                instead call raw_input() and wait for [Enter]""")
    else:
        __ACTIVE = True


def wait_failover(caller):
    if not ROOT.gROOT.IsBatch():
        log.warning(
            "{0} is failing over to raw_input()".format(caller.__name__))
        raw_input("press [Enter] to continue")


def start_new_gui_thread():
    """
    Attempt to start a new GUI thread, if possible.

    It is only possible to start one if there was one running on module import.
    """
    PyGUIThread = getattr(ROOT, 'PyGUIThread', None)

    if PyGUIThread is not None:
        assert not PyGUIThread.isAlive(), "GUI thread already running!"

    assert _processRootEvents, (
        "GUI thread wasn't started when rootwait was imported, "
        "so it can't be restarted")

    ROOT.keeppolling = 1
    ROOT.PyGUIThread = threading.Thread(
        None, _processRootEvents, None, (ROOT,))

    ROOT.PyGUIThread.finishSchedule = _finishSchedule
    ROOT.PyGUIThread.setDaemon(1)
    ROOT.PyGUIThread.start()
    log.debug("successfully started a new GUI thread")


def stop_gui_thread():
    """
    Try to stop the GUI thread. If it was running returns True,
    otherwise False.
    """
    PyGUIThread = getattr(ROOT, 'PyGUIThread', None)

    if PyGUIThread is None or not PyGUIThread.isAlive():
        log.debug("no existing GUI thread is runnng")
        return False

    ROOT.keeppolling = 0
    try:
        PyGUIThread.finishSchedule()
    except AttributeError:
        log.debug("unable to call finishSchedule() on PyGUIThread")
        pass
    PyGUIThread.join()
    log.debug("successfully stopped the existing GUI thread")
    return True


def get_visible_canvases():
    """
    Return a list of active GUI canvases
    (as opposed to invisible Batch canvases)
    """
    try:
        return [c for c in ROOT.gROOT.GetListOfCanvases() if not c.IsBatch()]
    except AttributeError:
        # We might be exiting and ROOT.gROOT will raise an AttributeError
        return []


def run_application_until_done():

    had_gui_thread = stop_gui_thread()

    ROOT.gApplication._threaded = True
    ROOT.gApplication.Run(True)

    if had_gui_thread:
        start_new_gui_thread()


def dispatcher(f):
    disp = ROOT.TPyDispatcher(f)
    keepalive(disp, f)
    return disp


def wait_for_zero_canvases(middle_mouse_close=False):
    """
    Wait for all canvases to be closed, or CTRL-c.

    If `middle_mouse_close`, middle click will shut the canvas.

    incpy.ignore
    """
    if not __ACTIVE:
        wait_failover(wait_for_zero_canvases)
        return

    @dispatcher
    def count_canvases():
        """
        Count the number of active canvases and finish gApplication.Run()
        if there are none remaining.

        incpy.ignore
        """
        if not get_visible_canvases():
            try:
                ROOT.gSystem.ExitLoop()
            except AttributeError:
                # We might be exiting and ROOT.gROOT will raise an AttributeError
                pass

    @dispatcher
    def exit_application_loop():
        """
        Signal handler for CTRL-c to cause gApplication.Run() to finish.

        incpy.ignore
        """
        ROOT.gSystem.ExitLoop()

    # Handle CTRL-c
    sh = ROOT.TSignalHandler(ROOT.kSigInterrupt, True)
    sh.Add()
    sh.Connect("Notified()", "TPyDispatcher",
               exit_application_loop, "Dispatch()")

    visible_canvases = get_visible_canvases()

    for canvas in visible_canvases:
        log.debug("waiting for canvas {0} to close".format(canvas.GetName()))
        canvas.Update()

        if middle_mouse_close:
            attach_event_handler(canvas)

        if not getattr(canvas, "_py_close_dispatcher_attached", False):
            # Attach a handler only once to each canvas
            canvas._py_close_dispatcher_attached = True
            canvas.Connect("Closed()", "TPyDispatcher",
                           count_canvases, "Dispatch()")
            keepalive(canvas, count_canvases)

    if visible_canvases and not ROOT.gROOT.IsBatch():
        run_application_until_done()

        # Disconnect from canvases
        for canvas in visible_canvases:
            if getattr(canvas, "_py_close_dispatcher_attached", False):
                canvas._py_close_dispatcher_attached = False
                canvas.Disconnect("Closed()", count_canvases, "Dispatch()")

wait = wait_for_zero_canvases


def wait_for_frame(frame):
    """
    wait until a TGMainFrame is closed or ctrl-c
    """
    if not frame:
        # It's already closed or maybe we're in batch mode
        return

    @dispatcher
    def close():
        ROOT.gSystem.ExitLoop()

    if not getattr(frame, "_py_close_dispatcher_attached", False):
        frame._py_close_dispatcher_attached = True
        frame.Connect("CloseWindow()", "TPyDispatcher", close, "Dispatch()")

    @dispatcher
    def exit_application_loop():
        """
        Signal handler for CTRL-c to cause gApplication.Run() to finish.

        incpy.ignore
        """
        ROOT.gSystem.ExitLoop()

    # Handle CTRL-c
    sh = ROOT.TSignalHandler(ROOT.kSigInterrupt, True)
    sh.Add()
    sh.Connect("Notified()", "TPyDispatcher",
               exit_application_loop, "Dispatch()")

    if not ROOT.gROOT.IsBatch():
        run_application_until_done()
        # Need to disconnect to prevent close handler from running when python
        # teardown has already commenced.
        frame.Disconnect("CloseWindow()", close, "Dispatch()")


def wait_for_browser_close(b):
    """
    Can be used to wait until a TBrowser is closed
    """
    if b:
        if not __ACTIVE:
            wait_failover(wait_for_browser_close)
            return
        wait_for_frame(b.GetBrowserImp().GetMainFrame())


def prevent_close_with_canvases():
    """
    Register a handler which prevents python from exiting until
    all canvases are closed
    """
    register(wait_for_zero_canvases)
