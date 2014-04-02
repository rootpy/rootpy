# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from .. import QROOT
from ..decorators import snake_case_methods
from .base import Plottable

__all__ = [
    'Line',
    'Ellipse',
    'Arrow',
]


@snake_case_methods
class Line(Plottable, QROOT.TLine):
    _ROOT = QROOT.TLine

    def __init__(self, *args, **kwargs):
        super(Line, self).__init__(*args)
        self._post_init(**kwargs)


@snake_case_methods
class Ellipse(Plottable, QROOT.TEllipse):
    _ROOT = QROOT.TEllipse

    def __init__(self, *args, **kwargs):
        super(Ellipse, self).__init__(*args)
        self._post_init(**kwargs)


@snake_case_methods
class Arrow(QROOT.TArrow):
    _ROOT = QROOT.TArrow
