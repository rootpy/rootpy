# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import
import sys

IN_NOSETESTS = False
if sys.argv and sys.argv[0].endswith('nosetests'):
    IN_NOSETESTS = True

IN_IPYTHON = '__IPYTHON__' in __builtins__
if IN_IPYTHON:
    try:
        # try to import OutStream from ipykernel if possible
        try:
            from ipykernel.iostream import OutStream
        except ImportError:
            from IPython.kernel.zmq.iostream import OutStream
        IN_IPYTHON_NOTEBOOK = isinstance(sys.stdout, OutStream)
    except ImportError: # pyzmq not installed?
        IN_IPYTHON_NOTEBOOK = False
else:
    IN_IPYTHON_NOTEBOOK = False

from collections import namedtuple

# DO NOT expose ROOT at module level here since that conflicts with rootpy.ROOT
# See issue https://github.com/rootpy/rootpy/issues/343
import ROOT as R

from .extern.six import string_types
from .logger import log
# Needed for "from rootpy import QROOT" by other modules
from .utils import quickroot as QROOT
from . import defaults
from .base import Object
from .info import __version__

__all__ = [
    'log',
    'ROOT_VERSION',
    'QROOT',
    'asrootpy',
    'lookup',
    'lookup_by_name',
    'lookup_rootpy',
    'register',
    'create',
]


class ROOTVersion(namedtuple('_ROOTVersionBase',
                             ['major', 'minor', 'micro'])):

    def __new__(cls, version):
        if version < 1E4:
            raise ValueError(
                "{0:d} is not a valid ROOT version integer".format(version))
        return super(ROOTVersion, cls).__new__(
            cls,
            int(version / 1E4), int((version / 1E2) % 100), int(version % 100))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{0:d}.{1:02d}/{2:02d}'.format(*self)


# Note: requires defaults import
ROOT_VERSION = ROOTVersion(QROOT.gROOT.GetVersionInt())
log.debug("Using ROOT {0}".format(ROOT_VERSION))


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

    'TList': 'collection.List',
    'TObjArray': 'collection.ObjArray',

    'TTree': 'tree.tree.Tree',
    'TNtuple': 'tree.tree.Ntuple',

    'TKey': 'io.file.Key',
    'TDirectoryFile': 'io.file.Directory',
    'TFile': 'io.file.File',
    'TMemFile': 'io.file.MemFile',

    'TStyle': 'plotting.style.Style',
    'TCanvas': 'plotting.canvas.Canvas',
    'TPad': 'plotting.canvas.Pad',
    'TPave': 'plotting.box.Pave',
    'TPaveStats': 'plotting.box.PaveStats',
    'TLegend': 'plotting.legend.Legend',
    'TEllipse': 'plotting.shapes.Ellipse',
    'TLine': 'plotting.shapes.Line',
    'TArrow': 'plotting.shapes.Arrow',

    'TF1': 'plotting.func.F1',
    'TF2': 'plotting.func.F2',
    'TF3': 'plotting.func.F3',

    'TGraph': ('plotting.graph.Graph', dict(type='default')),
    'TGraphErrors': ('plotting.graph.Graph', dict(type='errors')),
    'TGraphAsymmErrors': ('plotting.graph.Graph', dict(type='asymm')),
    'TGraphBentErrors': ('plotting.graph.Graph', dict(type='benterrors')),

    'TGraph2D': ('plotting.graph.Graph2D', dict(type='default')),
    'TGraph2DErrors': ('plotting.graph.Graph2D', dict(type='errors')),

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

    'TEfficiency': 'plotting.hist.Efficiency',

    'THStack': 'plotting.hist.HistStack',

    'TAxis': 'plotting.axis.Axis',

    'TVector2': 'vector.Vector2',
    'TVector3': 'vector.Vector3',
    'TLorentzVector': 'vector.LorentzVector',
    'TRotation': 'vector.Rotation',
    'TLorentzRotation': 'vector.LorentzRotation',

    'TMatrixT<float>': ('matrix.Matrix', dict(type='float')),
    'TMatrixT<double>': ('matrix.Matrix', dict(type='double')),
    'TMatrixTSym<float>': ('matrix.SymmetricMatrix', dict(type='float')),
    'TMatrixTSym<double>': ('matrix.SymmetricMatrix', dict(type='double')),

    'RooWorkspace': 'stats.workspace.Workspace',
    'RooStats::ModelConfig': 'stats.modelconfig.ModelConfig',
    'RooArgSet': 'stats.collection.ArgSet',
    'RooArgList': 'stats.collection.ArgList',
    'RooRealVar': 'stats.value.RealVar',
    'RooSimultaneous': 'stats.pdf.Simultaneous',
    'RooAddPdf': 'stats.pdf.AddPdf',
    'RooProdPdf': 'stats.pdf.ProdPdf',
    'RooCatType': 'stats.category.CatType',
    'RooCategory': 'stats.category.Category',
    'RooDataSet': 'stats.dataset.DataSet',
    'RooMinimizer': 'stats.fit.Minimizer',
    'RooFitResult': 'stats.fit.FitResult',

    'RooStats::HistFactory::Data': 'stats.histfactory.Data',
    'RooStats::HistFactory::Sample': 'stats.histfactory.Sample',
    'RooStats::HistFactory::HistoSys': 'stats.histfactory.HistoSys',
    'RooStats::HistFactory::HistoFactor': 'stats.histfactory.HistoFactor',
    'RooStats::HistFactory::OverallSys': 'stats.histfactory.OverallSys',
    'RooStats::HistFactory::NormFactor': 'stats.histfactory.NormFactor',
    'RooStats::HistFactory::ShapeSys': 'stats.histfactory.ShapeSys',
    'RooStats::HistFactory::ShapeFactor': 'stats.histfactory.ShapeFactor',
    'RooStats::HistFactory::Channel': 'stats.histfactory.Channel',
    'RooStats::HistFactory::Measurement': 'stats.histfactory.Measurement',
}

# map rootpy name to location in rootpy (i.e. Axis -> plotting.axis)
INIT_REGISTRY_ROOTPY = {}
for rtype, rptype in INIT_REGISTRY.items():
    if isinstance(rptype, tuple):
        rptype = rptype[0]
    cls_path, _, cls_name = rptype.rpartition('.')
    INIT_REGISTRY_ROOTPY[cls_name] = cls_path

# these dicts are populated as classes are registered at runtime
# ROOT class name -> rootpy class
REGISTRY = {}
# rootpy class name -> rootpy class
REGISTRY_ROOTPY = {}


def asrootpy(thing, **kwargs):
    # is this thing already converted?
    if isinstance(thing, Object):
        return thing

    warn = kwargs.pop('warn', False)
    after_init = kwargs.pop('after_init', False)

    # is this thing a class?
    if isinstance(thing, QROOT.PyRootType):
        if issubclass(thing, Object):
            return thing
        result = lookup(thing)
        if result is None:
            if warn:
                log.warn(
                    "There is no rootpy implementation "
                    "of the class `{0}`".format(thing.__name__))
            return thing
        if after_init:
            # preserve ROOT's __init__
            class asrootpy_cls(result):
                def __new__(self, *args, **kwargs):
                    return asrootpy(thing(*args, **kwargs), warn=warn)
            asrootpy_cls.__name__ = '{0}_asrootpy'.format(thing.__name__)
            return asrootpy_cls
        return result

    thing_cls = thing.__class__
    rootpy_cls = lookup(thing_cls)
    if rootpy_cls is None:
        if warn:
            log.warn(
                "A subclass of `{0}` is not "
                "implemented in rootpy".format(
                    thing_cls.__name__))
        return thing

    # cast
    thing.__class__ = rootpy_cls
    # call the _post_init if one exists
    if hasattr(thing, '_post_init'):
        thing._post_init(**kwargs)

    return thing


def _get_class(path, name):
    rootpy_module = __import__(
        'rootpy.' + path, globals(), locals(), [name], 0)
    return getattr(rootpy_module, name)


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
    elif isinstance(entry, string_types):
        path = entry
        dynamic_kwargs = None
    cls_path, _, rootpy_cls_name = path.rpartition('.')

    rootpy_cls = _get_class(cls_path, rootpy_cls_name)

    if dynamic_kwargs is not None:
        rootpy_cls = rootpy_cls.dynamic_cls(**dynamic_kwargs)
    REGISTRY[cls_name] = rootpy_cls
    return rootpy_cls


def lookup_rootpy(rootpy_cls_name):
    rootpy_cls = REGISTRY_ROOTPY.get(rootpy_cls_name, None)
    if rootpy_cls is not None:
        return rootpy_cls
    cls_path = INIT_REGISTRY_ROOTPY.get(rootpy_cls_name, None)
    if cls_path is None:
        return None

    rootpy_cls = _get_class(cls_path, rootpy_cls_name)

    REGISTRY_ROOTPY[rootpy_cls_name] = rootpy_cls
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
                log.debug(
                    "duplicate registration of "
                    "class `{0}`".format(name))
            REGISTRY[name] = cls
        return cls


def create(cls_name, *args, **kwargs):
    cls = getattr(R, cls_name, None)
    if cls is None:
        return None
    try:
        obj = cls(*args, **kwargs)
        return asrootpy(obj)
    except TypeError:
        return None
