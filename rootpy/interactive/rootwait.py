"""
Keeps root alive until ctrl-c is pressed or all canvases are closed
"""

import ROOT as R

exiting = False

@R.TPyDispatcher
def count_canvases():
    'incpy.ignore'
    global exiting
    if not exiting and not get_visible_canvases():
        exiting = True
        R.gSystem.ExitLoop()
        
@R.TPyDispatcher
def terminate():
    'incpy.ignore'
    global exiting
    exiting = True
    print " caught, will stop waiting"
    R.gSystem.ExitLoop()

def get_visible_canvases():
    'incpy.ignore'
    return [c for c in R.gROOT.GetListOfCanvases() if not c.IsBatch()]

def wait_for_zero_canvases():
    "Wait for all canvases to be closed, or ctrl-c incpy.ignore"
    
    global exiting
    exiting = False
    
    # Handle ctrl-c
    sh = R.TSignalHandler(R.kSigInterrupt, True)
    sh.Add()
    #sh.Connect("Notified()", "TApplication", R.gApplication, "Terminate()")
    #sh.Connect("Notified()", "TROOT",        R.gROOT,        "SetInterrupt()")
    sh.Connect("Notified()", "TPyDispatcher", terminate, "Dispatch()")
    
    visible_canvases = get_visible_canvases()
    
    global count_canvases
            
    for canvas in visible_canvases:
        if not getattr(canvas, "_py_close_dispatcher_attached", False):
            canvas._py_close_dispatcher_attached = True
            canvas.Connect("Closed()", "TPyDispatcher",
                           count_canvases, "Dispatch()")
                           
        canvas.Update()
        
    if visible_canvases and not R.gROOT.IsBatch():
        print "There are canvases open. Waiting until exit."
        R.gApplication.Run(True)
        exiting = True

wait = wait_for_zero_canvases

def prevent_close_with_canvases():
    register(wait_for_zero_canvases)

