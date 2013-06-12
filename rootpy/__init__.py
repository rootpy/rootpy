# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

# First import
import warnings
# show deprecation warnings
warnings.filterwarnings('default', category=DeprecationWarning)

from .logger import log

# Needed for "from rootpy import QROOT" by other modules
from .util import quickroot as QROOT
from . import defaults
from .core import Object
from .info import __version_info__, __version__

import ROOT

# Note: requires defaults import
ROOT_VERSION = QROOT.gROOT.GetVersionInt()
ROOT_VERSION_STR = QROOT.gROOT.GetVersion()
log.debug("Using ROOT {0}".format(ROOT_VERSION_STR))


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

"""
All rootpy wrappers are registered below. This dict maps the ROOT class name to
the path to the rootpy class or a tuple of both the path and keyword arguments
used in the dynamic_cls classmethod (see the ``Hist``, ``Hist2D`` and
``Hist3D`` classes in ``plotting.hist``).

This way rootpy is aware of all classes that inherit from ROOT classes without
needing to import everything when rootpy is first imported. This registry is
required to cast ROOT objects into the rootpy form when extracted from a ROOT
TFile, for example.
"""
INIT_REGISTRY = {
    
    'TList': 'root_collections.List',
    'TObjArray': 'root_collections.ObjArray',
    
    'TTree': 'tree.tree.Tree',
    'TNtuple': 'tree.tree.Ntuple',

    'TDirectoryFile': 'io.file.Directory',
    'TFile': 'io.file.File',

    'TStyle': 'plotting.style.Style',
    'TCanvas': 'plotting.canvas.Canvas',
    'TPad': 'plotting.canvas.Pad',
    'TLegend': 'plotting.legend.Legend',
    'TEllipse': 'plotting.shapes.Ellipse',
    'TLine': 'plotting.shapes.Line',

    'TGraphAsymmErrors': 'plotting.graph.Graph',
    'TGraph2D': 'plotting.graph.Graph2D',

    'TProfile': 'plotting.profile.Profile',
    'TProfile2D': 'plotting.profile.Profile2D',
    'TProfile3D': 'plotting.profile.Profile3D',

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

    'TVector2': 'math.physics.vector.Vector2',
    'TVector3': 'math.physics.vector.Vector3',
    'TLorentzVector': 'math.physics.vector.LorentzVector',
    'TRotation': 'math.physics.vector.Rotation',
    'TLorentzRotation': 'math.physics.vector.LorentzRotation',
}

if ROOT_VERSION >= 52800:
    INIT_REGISTRY['TEfficiency'] = 'plotting.hist.Efficiency'


# this dict is populated as classes are registered at runtime
REGISTRY = {}


def asrootpy(thing, **kwargs):

    # is this thing already converted?
    if isinstance(thing, Object):
        return thing

    warn = kwargs.pop("warn", True)

    # is this thing a class?
    if isinstance(thing, QROOT.PyRootType):
        if issubclass(thing, Object):
            return thing
        result = lookup(thing)
        if result is None:
            if warn:
                log.warn("There is no rootpy implementation of the class '{0}'"
                         .format(thing.__name__))
            return thing
        return result
    
    thing_cls = thing.__class__
    rootpy_cls = lookup(thing_cls)
    if rootpy_cls is None:
        if warn:
            log.warn("a subclass of %s is not implemented in rootpy" %
                    thing_cls.__name__)
        return thing

    # cast
    thing.__class__ = rootpy_cls
    # call the _post_init if one exists
    if hasattr(thing, '_post_init'):
        thing._post_init(**kwargs)

    return thing


def lookup(cls):

    cls_name = cls.__name__
    return lookup_by_name(cls_name)


def lookup_by_name(cls_name):

    if cls_name in REGISTRY:
        return REGISTRY[cls_name]
    if cls_name not in INIT_REGISTRY:
        return None
    entry = INIT_REGISTRY[cls_name]
    if isinstance(entry, tuple):
        path, dynamic_kwargs = entry
    elif isinstance(entry, basestring):
        path = entry
        dynamic_kwargs = None
    path_tokens = path.split('.')
    path, rootpy_cls_name = '.'.join(path_tokens[:-1]), path_tokens[-1]
    rootpy_module = __import__(
            path, globals(), locals(), [rootpy_cls_name], -1)
    rootpy_cls = getattr(rootpy_module, rootpy_cls_name)
    if dynamic_kwargs is not None:
        rootpy_cls = rootpy_cls.dynamic_cls(**dynamic_kwargs)
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

        if issubclass(cls, Object):
            # all rootpy classes which inherit from ROOT classes
            # must place the ROOT base class as the last class
            # in the inheritance list and inherit from Object
            rootbase = cls.__bases__[-1]
            cls_names = [rootbase.__name__]
        else:
            cls_names = [cls.__name__]

        if self.names is not None:
            cls_names += self.names

        for name in cls_names:
            if name in REGISTRY:
                log.warn("duplicate registration of class %s" % name)
            REGISTRY[name] = cls
        return cls


def create(cls_name, *args, **kwargs):

    cls = getattr(ROOT, cls_name, None)
    if cls is None:
        return None
    obj = cls(*args, **kwargs)
    return asrootpy(obj)


def gDirectory():
    # handle versions of ROOT older than 5.32.00
    if ROOT_VERSION < 53200:
        return ROOT.gDirectory
    else:
        return ROOT.gDirectory.func()
