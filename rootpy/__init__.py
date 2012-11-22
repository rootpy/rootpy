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
INIT_REGISTRY = {
    'TTree': 'tree.tree.Tree',

    'TDirectoryFile': 'io.file.Directory',

    'TCanvas': 'plotting.canvas.Canvas',
    'TPad': 'plotting.canvas.Pad',
    'TLegend': 'plotting.legend.Legend',

    'TGraphAsymmErrors': 'plotting.graph.Graph',
    'TGraph2D': 'plotting.graph.Graph2D',

    'TH1C': ('plotting.hist.Hist', dict(type='C')),
    'TH1S': ('plotting.hist.Hist', dict(type='S')),
    'TH1I': ('plotting.hist.Hist', dict(type='I')),
    'TH1F': ('plotting.hist.Hist', dict(type='F')),
    'TH1D': ('plotting.hist.Hist', dict(type='D')),

    'TH2C': ('plotting.hist.Hist2D', dict(type='C')),
    'TH2S': ('plotting.hist.Hist2D', dict(type='S')),
    'TH2I': ('plotting.hist.Hist2D', dict(type='I')),
    'TH2F': ('plotting.hist.Hist2D', dict(type='F')),
    'TH2D': ('plotting.hist.Hist2D', dict(type='D')),

    'TH3C': ('plotting.hist.Hist3D', dict(type='C')),
    'TH3S': ('plotting.hist.Hist3D', dict(type='S')),
    'TH3I': ('plotting.hist.Hist3D', dict(type='I')),
    'TH3F': ('plotting.hist.Hist3D', dict(type='F')),
    'TH3D': ('plotting.hist.Hist3D', dict(type='D')),

    'THStack': 'plotting.hist.HistStack',

    'TVector2': 'math.physics.vector.TVector2',
    'TVector3': 'math.physics.vector.TVector3',
    'TLorentzVector': 'math.physics.vector.LorentzVector',
    'TRotation': 'math.physics.vector.Rotation',
    'TLorentzRotation': 'math.physics.vector.LorentzRotation',
}

if ROOT.gROOT.GetVersionInt() ≥ 52800:
    INIT_REGISTRY['TEfficiency'] = 'plotting.hist.Efficiency'


# this dict is populated as classes are registered at runtime
REGISTRY = {}


def asrootpy(thing, **kwargs):

    # is this thing already converted?
    if isinstance(thing, Object):
        return thing

    thing_cls = thing.__class__
    rootpy_cls = lookup(thing_cls)
    if rootpy_cls is None:
        # warn that this class is not wrapped in rootpy
        # return the original
        warnings.warn(
                "A subclass of ``%s`` is not implemented in rootpy. "
                "Returning the original object instead." % thing_cls)
        return thing

    # cast
    thing.__class__ = rootpy_cls
    # call the _post_init if one exists
    if hasattr(thing, '_post_init'):
        thing._post_init(**kwargs)

    return thing


def lookup(cls):

    cls_name = cls.__name__
    if cls_name in REGISTRY:
        return REGISTRY[cls_name]
    if cls_name not in INIT_REGISTRY:
        return None
    entry = INIT_REGISTRY[cls_name]
    if isinstance(entry, tuple):
        path, init_kwargs = entry
    elif isinstance(entry, basestring):
        path = entry
        init_kwargs = {}
    path_tokens = path.split('.')
    path, rootpy_cls_name = path_tokens[:-1], path_tokens[-1]
    rootpy_module = __import__(
            path, globals(), locals(), [rootpy_cls_name], -1)
    rootpy_cls = getattr(rootpy_module, rootpy_cls_name)
    REGISTRY[cls_name] = rootpy_cls
    return rootpy_cls


class register(object):

    def __init__(self, names=None, builtin=False):

        if names is not None:
            if type(names) not in (list, tuple):
                names = [names]
        self.names = names
        self.builtin = builtin

    def __call__(self, cls):

        if self.builtin:
            cls_names = [cls.__name__]
        else:
            # all rootpy classes which inherit from ROOT classes
            # must place the ROOT base class as the last class
            # in the inheritance list
            rootbase = cls.__bases__[-1]
            cls_names = [rootbase.__name__]

        if self.names is not None:
            cls_names += self.names

        for name in cls_names:
            if name in REGISTRY:
                warnings.warn("Duplicate registration of class %s" % name)
            REGISTRY[name] = cls
        return cls


def create(cls_name, *args, **kwargs):

    try:
        cls = getattr(ROOT, cls_name)
        obj = cls(*args, **kwargs)
        return asrootpy(obj)
    except:
        return None
