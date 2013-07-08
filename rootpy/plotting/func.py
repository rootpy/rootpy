# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import QROOT
from ..decorators import snake_case_methods
from .core import Plottable
from ..core import NameOnlyObject


__all__ = [
    'F1',
    'F2',
    'F3',
]


@snake_case_methods
class F1(Plottable, NameOnlyObject, QROOT.TF1):

    def __init__(self, *args, **kwargs):

        name = kwargs.pop('name', None)
        super(F1, self).__init__(*args, name=name)
        self._post_init(**kwargs)


@snake_case_methods
class F2(Plottable, NameOnlyObject, QROOT.TF2):

    def __init__(self, *args, **kwargs):

        name = kwargs.pop('name', None)
        super(F2, self).__init__(*args, name=name)
        self._post_init(**kwargs)


@snake_case_methods
class F3(Plottable, NameOnlyObject, QROOT.TF3):

    def __init__(self, *args, **kwargs):

        name = kwargs.pop('name', None)
        super(F3, self).__init__(*args, name=name)
        self._post_init(**kwargs)
