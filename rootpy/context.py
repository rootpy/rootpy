from contextlib import contextmanager

import ROOT

@contextmanager
def preserve_current_canvas():
    """
    Context manager which ensures that the current canvas remains the current
    canvas when the context is left.
    """
    
    old = ROOT.gPad.func()
    try:
        yield
    finally:
        if old:
            old.cd()
        else:
            # Is it possible to set ROOT.gPad back to None, somehow?
            pass

@contextmanager
def preserve_batch_state():
    """
    Context manager which ensures the batch state is the same on exit as it was
    on entry.
    """

    old = ROOT.gROOT.IsBatch()
    try:
        yield
    finally:
        ROOT.gROOT.SetBatch(old)

@contextmanager
def invisible_canvas():
    """
    Context manager yielding a temporary canvas drawn in batch mode, invisible
    to the user. Original state is restored on exit.
    
    Example use; obtain X axis object without interfering with anything:
    
        with invisible_canvas() as c:
            efficiency.Draw()
            g = efficiency.GetPaintedGraph()
            return g.GetXaxis()
    """
    
    with preserve_batch_state():
        ROOT.gROOT.SetBatch()
        with preserve_current_canvas():
            c = ROOT.TCanvas()
            try:
                c.cd()
                yield c
            finally:
                c.Close()
