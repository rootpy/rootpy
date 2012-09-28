import ROOT
from ..core import snake_case_methods
from .core import Plottable


@snake_case_methods
class Ellipse(Plottable, ROOT.TEllipse):

    def __init__(self, *args, **kwargs):

        ROOT.TEllipse.__init__(self, *args)
        Plottable.__init__(self)
        self.decorate(**kwargs)
