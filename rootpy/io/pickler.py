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
  the ROOT file open. Pass use_proxy=False to disable this behavior.

"""
from __future__ import absolute_import

import sys
if sys.version_info[0] < 3:
    from cStringIO import StringIO
else:
    from io import StringIO

# need subclassing ability in 2.x
import pickle

import ROOT

from . import log; log = log[__name__]
from . import root_open
from ..context import preserve_current_directory
from ..extern.six import string_types


__all__ = [
    'dump',
    'load',
    'compat_hooks',
]


_compat_hooks = None
xdict = {}
xserial = 0

"""
Argh!  We can't store NULs in TObjStrings.
But pickle protocols > 0 are binary protocols, and will get corrupted
if we truncate at a NUL.
So, when we save the pickle data, make the mappings:

   0x00 -> 0xff 0x01
   0xff -> 0xff 0xfe

"""


def _protect(s):
    return s.replace(b'\377', b'\377\376').replace(b'\000', b'\377\001')


def _restore(s):
    return s.replace(b'\377\001', b'\000').replace(b'\377\376', b'\377')


class IO_Wrapper:
    def __init__(self):
        return self.reopen()

    def write(self, s):
        return self.__s.write(_protect(s).decode('utf-8'))

    def read(self, i):
        return self.__s.read(i).encode('utf-8')

    def readline(self):
        return self.__s.readline().encode('utf-8')

    def getvalue(self):
        return self.__s.getvalue()

    def setvalue(self, s):
        self.__s = StringIO(_restore(s.encode('utf-8')).decode('utf-8'))
        return

    def reopen(self):
        self.__s = StringIO()
        return


class ROOT_Proxy:
    def __init__(self, f, pid):
        self.__f = f
        self.__pid = pid
        self.__o = None

    def __getattr__(self, a):
        if self.__o is None:
            log.debug("unpickler proxy reading {0}".format(self.__pid))
            self.__o = self.__f.Get(self.__pid)
            self.__o.__class__.__module__ = 'ROOT'
        return getattr(self.__o, a)

    def __obj(self):
        if self.__o is None:
            log.debug("unpickler proxy reading {0}".format(self.__pid))
            self.__o = self.__f.Get(self.__pid)
            self.__o.__class__.__module__ = 'ROOT'
        return self.__o


class Pickler(pickle.Pickler):
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
        self.__pmap = {}
        if sys.version_info[0] < 3:
            # 2.X old-style classobj
            pickle.Pickler.__init__(self, self.__io, proto)
        else:
            super(Pickler, self).__init__(self.__io, proto)

    def dump(self, obj, key=None):
        """Write a pickled representation of obj to the open TFile."""
        if key is None:
            key = '_pickle'
        with preserve_current_directory():
            self.__file.cd()
            if sys.version_info[0] < 3:
                pickle.Pickler.dump(self, obj)
            else:
                super(Pickler, self).dump(obj)
            s = ROOT.TObjString(self.__io.getvalue())
            self.__io.reopen()
            s.Write(key)
            self.__file.GetFile().Flush()
            self.__pmap.clear()

    def clear_memo(self):
        """Clears the pickler's internal memo."""
        self.__pickle.memo.clear()

    def persistent_id(self, obj):
        if hasattr(obj, '_ROOT_Proxy__obj'):
            obj = obj._ROOT_Proxy__obj()
        if isinstance(obj, ROOT.TObject):
            """
            Write the object, and return the resulting NAME;CYCLE.
            We used to do this::

               o.Write()
               k = self.__file.GetKey(o.GetName())
               pid = "{0};{1:d}".format(k.GetName(), k.GetCycle())

            It turns out, though, that destroying the python objects
            referencing the TKeys is quite expensive (O(logN) where N is the
            total number of pyroot objects?).  Although we want to allow for
            the case of saving multiple objects with the same name, the most
            common case is that the name has not already been written to the
            file.  So we optimize for that case, doing the key lookup before we
            write the object, not after.  (Note further: GetKey() is very slow
            if the key does not actually exist, as it does a linear search of
            the key list.  We use FindObject instead for the initial
            lookup, which is a hashed lookup, but it is not guaranteed to
            find the highest cycle.  So if we do find an existing key, we
            need to look up again using GetKey.
            """
            nm = obj.GetName()
            key = self.__keys.FindObject(nm)
            obj.Write()
            if key:
                key = self.__file.GetKey(nm)
                pid = '{0};{1:d}'.format(nm, key.GetCycle())
            else:
                pid = nm + ';1'
            return pid


class Unpickler(pickle.Unpickler):
    def __init__(self, root_file, use_proxy=True, use_hash=False):
        """Create a ROOT unpickler.
        `file` should be a ROOT TFile.
        """
        global xserial
        xserial += 1
        self.__use_proxy = use_proxy
        self.__file = root_file
        self.__io = IO_Wrapper()
        self.__n = 0
        self.__serial = '{0:d}-'.format(xserial).encode('utf-8')
        xdict[self.__serial] = root_file
        if sys.version_info[0] < 3:
            pickle.Unpickler.__init__(self, self.__io)
        else:
            super(Unpickler, self).__init__(self.__io)

        if use_hash:
            htab = {}
            ctab = {}
            for k in root_file.GetListOfKeys():
                nm = k.GetName()
                cy = k.GetCycle()
                htab[(nm, cy)] = k
                if cy > ctab.get(nm, 0):
                    ctab[nm] = cy
                    htab[(nm, 9999)] = k
            root_file._htab = htab
            oget = root_file.Get

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
                #ctx = ROOT.TDirectory.TContext(file)
                ret = ret.ReadObj()
                #del ctx
                return ret
            root_file.Get = xget

    def load(self, key=None):
        """Read a pickled object representation from the open file."""
        if key is None:
            key = '_pickle'
        obj = None
        if _compat_hooks:
            save = _compat_hooks[0]()
        try:
            self.__n += 1
            s = self.__file.Get(key + ';{0:d}'.format(self.__n))
            self.__io.setvalue(s.GetName())
            if sys.version_info[0] < 3:
                obj = pickle.Unpickler.load(self)
            else:
                obj = super(Unpickler, self).load()
            self.__io.reopen()
        finally:
            if _compat_hooks:
                save = _compat_hooks[1](save)
        return obj

    def persistent_load(self, pid):
        log.debug("unpickler reading {0}".format(pid))
        if self.__use_proxy:
            obj = ROOT_Proxy(self.__file, pid)
        else:
            obj = self.__file.Get(pid)
        xdict[self.__serial + pid] = obj
        return obj

    def find_class(self, module, name):
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

    # Python 2.x
    find_global = find_class


def compat_hooks(hooks):
    """Set compatibility hooks.
    If this is set, then hooks[0] is called before loading, and hooks[1] is
    called after loading.  hooks[1] is called with the return value of hooks[0]
    as an argument.  This is useful for backwards compatibility in some
    situations.
    """
    global _compat_hooks
    _compat_hooks = hooks


def dump(obj, root_file, proto=0, key=None):
    """Dump an object into a ROOT TFile.

    `root_file` may be an open ROOT file or directory, or a string path to an
    existing ROOT file.
    """
    if isinstance(root_file, string_types):
        root_file = root_open(root_file, 'recreate')
        own_file = True
    else:
        own_file = False
    ret = Pickler(root_file, proto).dump(obj, key)
    if own_file:
        root_file.Close()
    return ret


def load(root_file, use_proxy=True, key=None):
    """Load an object from a ROOT TFile.

    `root_file` may be an open ROOT file or directory, or a string path to an
    existing ROOT file.
    """
    if isinstance(root_file, string_types):
        root_file = root_open(root_file)
        own_file = True
    else:
        own_file = False
    obj = Unpickler(root_file, use_proxy).load(key)
    if own_file:
        root_file.Close()
    return obj
