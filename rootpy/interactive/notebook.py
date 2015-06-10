# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Register display formatters for ROOT objects when running in an interactive
IPython notebook.

Based on an implementation here: https://gist.github.com/mazurov/6194738
"""
import tempfile

from .. import IN_IPYTHON
from ..plotting import Canvas
from ..context import preserve_current_canvas
from ..utils.hook import classhook, super_overridden

if IN_IPYTHON:
    from IPython.core import display

__all__ = [
    'configure',
]


def _display_canvas(canvas):
    file_handle = tempfile.NamedTemporaryFile(suffix='.png')
    canvas.SaveAs(file_handle.name)
    ip_img = display.Image(filename=file_handle.name, format='png', embed=True)
    return ip_img._repr_png_()


def _draw_image(meth, *args, **kwargs):
    file_handle = tempfile.NamedTemporaryFile(suffix='.png')
    with preserve_current_canvas():
        canvas = Canvas()
        meth(*args, **kwargs)
        canvas.SaveAs(file_handle.name)
    return display.Image(filename=file_handle.name, format='png', embed=True)


def _display_any(obj):
    return _draw_image(obj.Draw)._repr_png_()


def configure():
    if not IN_IPYTHON:
        raise RuntimeError("not currently running in IPython")
    import ROOT
    # trigger PyROOT's finalSetup()
    ROOT.kTRUE
    # canvases will be displayed inline
    ROOT.gROOT.SetBatch()
    try:
        # only available if running in IPython:
        shell = get_ipython()
    except NameError:
        # must be in non-interactive mode (ipcluster?)
        return
    # register display functions with PNG formatter:
    png_formatter = shell.display_formatter.formatters['image/png']
    png_formatter.for_type(ROOT.TCanvas, _display_canvas)
    png_formatter.for_type(ROOT.TF1, _display_any)
    png_formatter.for_type(ROOT.TH1, _display_any)
    png_formatter.for_type(ROOT.THStack, _display_any)
    png_formatter.for_type(ROOT.TGraph, _display_any)
    png_formatter.for_type(ROOT.TGraph2D, _display_any)
