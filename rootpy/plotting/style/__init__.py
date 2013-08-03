# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from ... import log; log = log[__name__]
from ... import asrootpy, QROOT
from ...core import Object


def get_style(name, mpl=False):
    if mpl:
        try:
            module = __import__('{0}.style_mpl'.format(name.lower()),
                                globals(), locals(), ['STYLE'], -1)
            style = getattr(module, 'STYLE')
        except ImportError, AttributeError:
            raise ValueError(
                "matplotlib style '{0}' is not defined".format(name))
    else:
        # is the style already created?
        for s in ROOT.gROOT.GetListOfStyles():
            # make search case-insensitive
            if s.GetName().lower() == name.lower():
                return asrootpy(s)
        # if not then attempt to locate it in rootpy
        try:
            module = __import__('{0}.style'.format(name.lower()),
                                globals(), locals(), ['STYLE'], -1)
            style = getattr(module, 'STYLE')
        except ImportError, AttributeError:
            raise ValueError(
                "ROOT style '{0}' is not defined".format(name))
    return style


def set_style(style, mpl=False):
    """
    If mpl is False accept either style name or a TStyle instance.
    If mpl is True accept either style name or a matplotlib.rcParams-like
    dictionary
    """
    if mpl:
        import matplotlib as mpl

        style_dictionary = {}
        if isinstance(style, basestring):
            style_dictionary = get_style(style, mpl=True)
            log.info("using matplotlib style '{0}'".format(style))
        elif isinstance(style, dict):
            style_dictionary = style
            log.info("using user-defined matplotlib style")
        else:
            raise TypeError("style must be a matplotlib style name or dict")
        for k, v in style_dictionary.iteritems():
            mpl.rcParams[k] = v
    else:
        if isinstance(style, basestring):
            style = get_style(style)
        log.info("using ROOT style '{0}'".format(style.GetName()))
        style.cd()


class Style(Object, QROOT.TStyle):

    def __enter__(self):

        set_style(self)
        return self

    def __exit__(self, type, value, traceback):

        return False
