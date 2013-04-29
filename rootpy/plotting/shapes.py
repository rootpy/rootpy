# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from .. import QROOT
from ..decorators import snake_case_methods
from .core import Plottable


@snake_case_methods
class Ellipse(Plottable, QROOT.TEllipse):

    def __init__(self, *args, **kwargs):

        super(Ellipse, self).__init__(*args, **kwargs)

#TODO: add more shapes here
