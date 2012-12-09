import ROOT

from .. import QROOT
from ..core import snake_case_methods
from .core import Plottable


@snake_case_methods
class Ellipse(Plottable, QROOT.TEllipse):

    def __init__(self, *args, **kwargs):

        ROOT.TEllipse.__init__(self, *args)
        Plottable.__init__(self)
        self.decorate(**kwargs)

#TODO: add more shapes here
