# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .base import _StyleContainer
from ..extern.six import string_types

__all__ = [
    'Font',
]

fonts_root2text = {
    1: 'times-medium-i-normal',
    2: 'times-bold-r-normal',
    3: 'times-bold-i-normal',
    4: 'helvetica-medium-r-normal',
    5: 'helvetica-medium-o-normal',
    6: 'helvetica-bold-r-normal',
    7: 'helvetica-bold-o-normal',
    8: 'courier-medium-r-normal',
    9: 'courier-medium-o-normal',
    10: 'courier-bold-r-normal',
    11: 'courier-bold-o-normal',
    12: 'symbol-medium-r-normal',
    13: 'times-medium-r-normal',
    14: 'wingdings',
    15: 'symbol-italic',
    }

fonts_text2root = dict([
    (value, key) for key, value in fonts_root2text.items()])


class Font(_StyleContainer):

    def __init__(self, font, prec=3):
        self._input = font
        if isinstance(font, string_types):
            if font not in fonts_text2root:
                raise ValueError("font '{0}' is not understood".format(font))
            self._root = fonts_text2root[font]
        else:
            if font not in fonts_root2text:
                raise ValueError("font '{0}' is not understood".format(font))
            self._root = font
        self._root *= 10
        self._root += prec
        # conversion to mpl not implemented
        self._mpl = None


if __name__ == '__main__':
    # Example from http://root.cern.ch/root/html/TAttText.html#T5
    from rootpy.plotting import Canvas
    from rootpy.interactive import wait
    from ROOT import TLatex

    c = Canvas(500, 700, name="ROOT Fonts", title="ROOT Fonts")
    c.Range(0, 0, 1, 1)
    c.SetBorderSize(2)
    c.SetFrameFillColor(0)

    def get_text(x, y, f, s):
        t = TLatex(x, y, "#font[41]{{0:d} :} {1}".format(f(), s))
        t.SetTextFont(f('root'))
        t.SetTextAlign(12)
        t.SetTextSize(0.048)
        return t

    y = 0.95
    prec = 2
    for font in sorted(fonts_root2text.keys()):
        f = Font(font, prec)
        if font != 14:
            t = get_text(0.02, y, f, "ABCDEFGH abcdefgh 0123456789 @#$")
        else:
            t = get_text(0.02, y, f, "ABCD efgh 01234 @#$")
        t.Draw()
        y -= 0.065
    wait()
