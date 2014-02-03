# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
Register display formatters for ROOT objects when running in an interactive
IPython notebook.

Based on an implementation here: https://gist.github.com/mazurov/6194738
"""
import tempfile
if '__IPYTHON__' in __builtins__:
    from IPython.core import display

from ..plotting import Canvas
from ..context import preserve_current_canvas
from ..utils.hook import classhook, super_overridden

__all__ = [
    'configure',
]

DEFAULT_CANVAS = None


def _display_canvas(canvas):
    file_handle = tempfile.NamedTemporaryFile(suffix='.png')
    canvas.SaveAs(file_handle.name)
    ip_img = display.Image(filename=file_handle.name, format='png', embed=True)
    return ip_img._repr_png_()


def _draw_image(meth, *args, **kwargs):
    global DEFAULT_CANVAS
    if DEFAULT_CANVAS is None:
        DEFAULT_CANVAS = Canvas()
    file_handle = tempfile.NamedTemporaryFile(suffix='.png')
    with preserve_current_canvas():
        DEFAULT_CANVAS.cd()
        DEFAULT_CANVAS.Clear()
        meth(*args, **kwargs)
        DEFAULT_CANVAS.SaveAs(file_handle.name)
    return display.Image(filename=file_handle.name, format='png', embed=True)


def _display_any(obj):
    return _draw_image(obj.Draw)._repr_png_()


def configure():
    import ROOT
    # trigger PyROOT's finalSetup()
    ROOT.kTRUE
    # canvases will be displayed inline
    ROOT.gROOT.SetBatch()
    # only available if running in IPython:
    shell = get_ipython()
    # register display functions with PNG formatter:
    png_formatter = shell.display_formatter.formatters['image/png']
    png_formatter.for_type(ROOT.TCanvas, _display_canvas)
    png_formatter.for_type(ROOT.TF1, _display_any)
    png_formatter.for_type(ROOT.TH1, _display_any)
