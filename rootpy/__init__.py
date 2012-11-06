from .logger import log

from . import defaults
from .info import __version_info__, __version__

import warnings

# show deprecation warnings
warnings.filterwarnings('always', category=DeprecationWarning)

class ROOTError(RuntimeError):
    """
    Exception class representing a ROOT error/warning message.
    """
    def __init__(self, level, location, msg):
        self.level, self.location, self.msg = level, location, msg

    def __str__(self):
        return "level={0}, loc='{1}', msg='{2}'".format(
            self.level, self.location, self.msg)

def rootpy_source_dir():
    import rootpy
    from os.path import abspath, dirname
    from inspect import getfile
    from sys import modules
    path = dirname(getfile(modules[__name__]))
    absp = abspath(path)
    return path, absp

_ROOTPY_SOURCE_PATH, _ROOTPY_SOURCE_ABSPATH = rootpy_source_dir()
del rootpy_source_dir

'''
All rootpy wrappers are registered below. This dict maps the ROOT class name to
a 2 or 3-tuple of the path to the rootpy class, the rootpy class name, and
optionally arguments required by the __new__ method to properly set the base
class (see the Hist, Hist2D and Hist3D classes).

This way rootpy is "aware" of all classes within the package that wrap ROOT
classes without needing to import everything up front. This registry is required
to enable rootpy to "cast" ROOT objects into the rootpy form when extracted from
a ROOT TFile for example.
'''
REGISTRY = {
    'TTree': ('tree.tree', 'Tree',),

    'TDirectoryFile': ('io.file', 'Directory',),

    'TCanvas': ('plotting.canvas', 'Canvas',),
    'TPad': ('plotting.canvas', 'Pad',),
    'TLegend': ('plotting.legend', 'Legend',),

    'TGraphAsymmErrors': ('plotting.graph', 'Graph',),
    'TGraph2D': ('plotting.graph', 'Graph2D',),

    'TH1C': ('plotting.hist', 'Hist', dict(type='C')),
    'TH1S': ('plotting.hist', 'Hist', dict(type='S')),
    'TH1I': ('plotting.hist', 'Hist', dict(type='I')),
    'TH1F': ('plotting.hist', 'Hist', dict(type='F')),
    'TH1D': ('plotting.hist', 'Hist', dict(type='D')),

    'TH2C': ('plotting.hist', 'Hist2D', dict(type='C')),
    'TH2S': ('plotting.hist', 'Hist2D', dict(type='S')),
    'TH2I': ('plotting.hist', 'Hist2D', dict(type='I')),
    'TH2F': ('plotting.hist', 'Hist2D', dict(type='F')),
    'TH2D': ('plotting.hist', 'Hist2D', dict(type='D')),

    'TH3C': ('plotting.hist', 'Hist3D', dict(type='C')),
    'TH3S': ('plotting.hist', 'Hist3D', dict(type='S')),
    'TH3I': ('plotting.hist', 'Hist3D', dict(type='I')),
    'TH3F': ('plotting.hist', 'Hist3D', dict(type='F')),
    'TH3D': ('plotting.hist', 'Hist3D', dict(type='D')),
}
