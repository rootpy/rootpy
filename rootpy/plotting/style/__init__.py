# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from ... import log; log = log[__name__]
from ... import asrootpy

import ROOT
from ... import QROOT


def get_style(name):
    # is the style already created?
    style = asrootpy(ROOT.gROOT.GetStyle(name))
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

def set_style(style):
    """
    Accepts either style name or a TStyle instance
    """
    if isinstance(style, basestring):
        style = get_style(style)
    log.info("using style '{0}'".format(style.GetName()))
    style.cd()
    ROOT.gROOT.ForceStyle()


class Style(QROOT.TStyle):

    def __enter__(self):

        set_style(self)
        return self

    def __exit__(self, type, value, traceback):

        return False
