# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import QROOT
from ..decorators import snake_case_methods
from .core import Plottable


__all__ = [
    'F1',
    'F2',
    'F3',
]


@snake_case_methods
class F1(Plottable, QROOT.TF1):

    def __init__(self, *args, **kwargs):

        super(F1, self).__init__(*args)
        self._post_init(**kwargs)


@snake_case_methods
class F2(Plottable, QROOT.TF2):

    def __init__(self, *args, **kwargs):

        super(F2, self).__init__(*args)
        self._post_init(**kwargs)


@snake_case_methods
class F3(Plottable, QROOT.TF3):

    def __init__(self, *args, **kwargs):

        super(F3, self).__init__(*args)
        self._post_init(**kwargs)
