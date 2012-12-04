from contextlib import contextmanager
from ...context import preserve_current_style
from ... import log; log = log[__name__]

import ROOT


def get_style(name):
    # is the style already created?
    style = ROOT.gROOT.GetStyle(name)
    if style:
        return style
    # if not then attempt to locate it in rootpy
    try:
        module = __import__('%s.style' % name.lower(),
                globals(), locals(), ['STYLE'], -1)
        style = getattr(module, 'STYLE')
    except ImportError, AttributeError:
        raise ValueError("style '%s' is not defined" % name)
    return style

def set_style(name):

    style = get_style(name)
    log.info("using style '{0}'".format(name))
    ROOT.gROOT.SetStyle(style.GetName())
    ROOT.gROOT.ForceStyle()

@contextmanager
def using_style(name):
    """
    Context manager which switches to the style named 'name' and then switches
    back to the previous style when the context is left.
    """
    with preserve_current_style():
        set_style(name)
        yield
