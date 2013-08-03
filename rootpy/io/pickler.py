# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
# Original author: Scott Snyder scott.snyder(a)cern.ch, 2004.

"""Pickle python data into a ROOT file, preserving references to ROOT objects.

This module allows pickling python objects into a ROOT file. The python
objects may contain references to named ROOT objects. If one has set up a
structure of python objects to hold ROOT histograms, this provides a convenient
way of saving and restoring your histograms. The pickled python data are
stored in an additional string object in the ROOT file; any ROOT objects are
stored as usual. (Thus, ROOT files written by the pickler can be read just
like any other ROOT file if you don't care about the python data.)

Here's an example of writing a pickle::

   from rootpy.plotting import Hist
   from rootpy.io.pickler import dump
   hlist = []
   for i in range(10):
       hlist.append(Hist(10, 0, 10))
   dump(hlist, 'test.root')

This writes a list of histograms to test.root. The histograms may be read back
like this::

   from rootpy.io.pickler import load
   hlist = load('test.root')

The following additional notes apply:

* Pickling may not always work correctly for the case of python objects
  deriving from ROOT objects. It will probably also not work for the case of
  ROOT objects which do not derive from TObject.

* When the pickled data are being read, if a class doesn't exist,
  a dummy class with no methods will be used instead. This is different
  from the standard pickle behavior (where it would be an error), but it
  simplifies usage in the common case where the class is being used to hold
  histograms, and its methods are entirely concerned with filling the
  histograms.

* When restoring a reference to a ROOT object, the default behavior
  is to not read the ROOT object itself, but instead to create a proxy. The
  ROOT object will then be read the first time the proxy is accessed. This can
  help significantly with time and memory usage if you're only accessing a
  small fraction of the ROOT objects, but it does mean that you need to keep
  the ROOT file open. Pass use_proxy=0 to disable this behavior.

"""
from . import log; log = log[__name__]
from . import root_open
from ..context import preserve_current_directory

from cStringIO import StringIO
import cPickle
import ROOT
import sys


__all__ = [
    'dump',
    'load',
    'compat_hooks',
]


_compat_hooks = None
xdict = {}
xserial = 0


# Argh!  We can't store NULs in TObjStrings.
# But pickle protocols > 0 are binary protocols, and will get corrupted
# if we truncate at a NUL.
# So, when we save the pickle data, make the mappings:
#  0x00 -> 0xff 0x01
#  0xff -> 0xff 0xfe


def _protect(s):
    return s.replace('\377', '\377\376').replace('\000', '\377\001')


def _restore(s):
    return s.replace('\377\001', '\000').replace('\377\376', '\377')


class IO_Wrapper:
    def __init__(self):
        return self.reopen()

    def write(self, s):
        return self.__s.write(_protect(s))

    def read(self, i):
        return self.__s.read(i)

    def readline(self):
        return self.__s.readline()

    def getvalue(self):
        return self.__s.getvalue()

    def setvalue(self, s):
        self.__s = StringIO(_restore(s))
        return

    def reopen(self):
        self.__s = StringIO()
        return


class Pickler:
    def __init__(self, file, proto=0):
        """Create a root pickler.
        `file` should be a ROOT TFile. `proto` is the python pickle protocol
        version to use.  The python part will be pickled to a ROOT
        TObjString called _pickle; it will contain references to the
        ROOT objects.
        """
        self.__file = file
        self.__keys = file.GetListOfKeys()
        self.__io = IO_Wrapper()
        self.__pickle = cPickle.Pickler(self.__io, proto)
        self.__pickle.persistent_id = self._persistent_id
        self.__pmap = {}

    def dump(self, o, key=None):
        """Write a pickled representation of o to the open TFile."""
        if key is None:
            key = '_pickle'
        with preserve_current_directory():
            self.__file.cd()
            self.__pickle.dump(o)
            s = ROOT.TObjString(self.__io.getvalue())
            self.__io.reopen()
            s.Write(key)
            self.__file.GetFile().Flush()
            self.__pmap.clear()

    def clear_memo(self):
        """Clears the pickler's internal memo."""
        self.__pickle.memo.clear()

    def _persistent_id(self, o):
        if hasattr(o, '_ROOT_Proxy__obj'):
            o = o._ROOT_Proxy__obj()
        if isinstance(o, ROOT.TObject):
            # Write the object, and return the resulting NAME;CYCLE.
            # We used to to this like this:
            #o.Write()
            #k = self.__file.GetKey(o.GetName())
            #pid = "{0};{1:d}".format(k.GetName(), k.GetCycle())
            # It turns out, though, that destroying the python objects
            # referencing the TKeys is quite expensive (O(logN) where
            # N is the total number of pyroot objects?).  Although
            # we want to allow for the case of saving multiple objects
            # with the same name, the most common case is that the name
            # has not already been written to the file.  So we optimize
            # for that case, doing the key lookup before we write the
            # object, not after.  (Note further: GetKey() is very slow
            # if the key does not actually exist, as it does a linear
            # search of the key list.  We use FindObject instead for the
            # initial lookup, which is a hashed lookup, but it is not
            # guaranteed to find the highest cycle.  So if we do
            # find an existing key, we need to look up again using GetKey.
            nm = o.GetName()
            k = self.__keys.FindObject(nm)
            o.Write()
            if k:
                k = self.__file.GetKey(nm)
                pid = '{0};{1:d}'.format(nm, k.GetCycle())
            else:
                pid = nm + ';1'
            return pid


class ROOT_Proxy:
    def __init__(self, f, pid):
        self.__f = f
        self.__pid = pid
        self.__o = None

    def __getattr__(self, a):
        if self.__o is None:
            self.__o = self.__f.Get(self.__pid)
            if self.__o.__class__.__module__ != 'ROOT':
                self.__o.__class__.__module__ = 'ROOT'
        return getattr(self.__o, a)

    def __obj(self):
        if self.__o is None:
            self.__o = self.__f.Get(self.__pid)
            if self.__o.__class__.__module__ != 'ROOT':
                self.__o.__class__.__module__ = 'ROOT'
        return self.__o


class Unpickler:
    def __init__(self, file, use_proxy=True, use_hash=False):
        """Create a ROOT unpickler.
        `file` should be a ROOT TFile.
        """
        global xserial
        xserial += 1
        self.__use_proxy = use_proxy
        self.__file = file
        self.__io = IO_Wrapper()
        self.__unpickle = cPickle.Unpickler(self.__io)
        self.__unpickle.persistent_load = self._persistent_load
        self.__unpickle.find_global = self._find_class
        self.__n = 0
        self.__serial = '{0:d}-'.format(xserial)
        xdict[self.__serial] = file

        if use_hash:
            htab = {}
            ctab = {}
            for k in file.GetListOfKeys():
                nm = k.GetName()
                cy = k.GetCycle()
                htab[(nm, cy)] = k
                if cy > ctab.get(nm, 0):
                    ctab[nm] = cy
                    htab[(nm, 9999)] = k
            file._htab = htab
            oget = file.Get

            def xget(nm0):
                nm = nm0
                ipos = nm.find(';')
                if ipos >= 0:
                    cy = nm[ipos+1]
                    if cy == '*':
                        cy = 10000
                    else:
                        cy = int(cy)
                    nm = nm[:ipos - 1]
                else:
                    cy = 9999
                ret = htab.get((nm, cy), None)
                if not ret:
                    log.warning(
                        "did't find {0} {1} {2}".format(nm, cy, len(htab)))
                    return oget(nm0)
                #ctx = ROOT.TDirectory.TContext (file)
                ret = ret.ReadObj()
                #del ctx
                return ret
            file.Get = xget

    def load(self, key=None):
        """Read a pickled object representation from the open file."""
        if key is None:
            key = '_pickle'
        o = None
        if _compat_hooks:
            save = _compat_hooks[0]()
        try:
            self.__n += 1
            s = self.__file.Get(key + ';{0:d}'.format(self.__n))
            self.__io.setvalue(s.GetName())
            o = self.__unpickle.load()
            self.__io.reopen()
        finally:
            if _compat_hooks:
                save = _compat_hooks[1](save)
        return o

    def _persistent_load(self, pid):
        if self.__use_proxy:
            o = ROOT_Proxy(self.__file, pid)
        else:
            o = self.__file.Get(pid)
        log.debug("load {0} {1}".format(pid, o))
        xdict[self.__serial + pid] = o
        return o

    def _find_class(self, module, name):
        try:
            try:
                __import__(module)
                mod = sys.modules[module]
            except ImportError:
                log.info("Making dummy module {0}".format(module))

                class DummyModule:
                    pass

                mod = DummyModule()
                sys.modules[module] = mod
            klass = getattr(mod, name)
            return klass
        except AttributeError:
            log.info("Making dummy class {0}.{1}".format(module, name))
            mod = sys.modules[module]

            class Dummy(object):
                pass

            setattr(mod, name, Dummy)
            return Dummy


def compat_hooks(hooks):
    """Set compatibility hooks.
    If this is set, then hooks[0] is called before loading, and hooks[1] is
    called after loading.  hooks[1] is called with the return value of hooks[0]
    as an argument.  This is useful for backwards compatibility in some
    situations.
    """
    global _compat_hooks
    _compat_hooks = hooks


def dump(o, f, proto=0, key=None):
    """Dump object O to the ROOT TFile `f`.

    `f` may be an open ROOT file or directory, or a string path to an existing
    ROOT file.
    """
    if isinstance(f, basestring):
        f = root_open(f, 'recreate')
        own_file = True
    else:
        own_file = False
    ret = Pickler(f, proto).dump(o, key)
    if own_file:
        f.Close()
    return ret


def load(f, use_proxy=1, key=None):
    """Load an object from the ROOT TFile `f`.

    `f` may be an open ROOT file or directory, or a string path to an existing
    ROOT file.
    """
    if isinstance(f, basestring):
        f = root_open(f)
    return Unpickler(f, use_proxy).load(key)
