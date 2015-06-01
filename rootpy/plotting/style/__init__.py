# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import ROOT

from ... import log; log = log[__name__]
from ... import asrootpy, QROOT
from ...base import Object
from ...extern.six import string_types

__all__ = [
    'get_style',
    'set_style',
    'Style',
]


def _kwargs_to_name(name, **kwargs):
    if not kwargs:
        return name.upper()
    return '{0}({1})'.format(name.upper(), ', '.join([
        '='.join(map(str, item))
            for item in sorted(kwargs.items())]))


def get_style(name, mpl=False, **kwargs):
    if mpl:
        try:
            module = __import__(
                'rootpy.plotting.style.{0}.style_mpl'.format(name.lower()),
                globals(), locals(), ['STYLE'], 0)
            style_func = getattr(module, 'style_mpl')
        except (ImportError, AttributeError):
            raise ValueError(
                "matplotlib style '{0}' is not defined".format(name))
        style = style_func(**kwargs)
    else:
        # is the style already created?
        for s in ROOT.gROOT.GetListOfStyles():
            # make search case-insensitive
            if s.GetName().lower() == name.lower():
                return asrootpy(s)
        # if not then attempt to locate it in rootpy
        try:
            module = __import__(
                'rootpy.plotting.style.{0}.style'.format(name.lower()),
                globals(), locals(), ['style'], 0)
            style_func = getattr(module, 'style')
        except (ImportError, AttributeError):
            raise ValueError(
                "ROOT style '{0}' is not defined".format(name))
        name = _kwargs_to_name(name, **kwargs)
        style = style_func(name, **kwargs)
    return style


def set_style(style, mpl=False, **kwargs):
    """
    If mpl is False accept either style name or a TStyle instance.
    If mpl is True accept either style name or a matplotlib.rcParams-like
    dictionary
    """
    if mpl:
        import matplotlib as mpl

        style_dictionary = {}
        if isinstance(style, string_types):
            style_dictionary = get_style(style, mpl=True, **kwargs)
            log.info("using matplotlib style '{0}'".format(style))
        elif isinstance(style, dict):
            style_dictionary = style
            log.info("using user-defined matplotlib style")
        else:
            raise TypeError("style must be a matplotlib style name or dict")
        for k, v in style_dictionary.iteritems():
            mpl.rcParams[k] = v
    else:
        if isinstance(style, string_types):
            style = get_style(style, **kwargs)
        log.info("using ROOT style '{0}'".format(style.GetName()))
        style.cd()


class Style(Object, QROOT.TStyle):
    _ROOT = QROOT.TStyle

    def __enter__(self):
        set_style(self)
        return self

    def __exit__(self, type, value, traceback):
        return False
