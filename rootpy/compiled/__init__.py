"""
#include <string>

namespace PyROOT { namespace Utility { const std::string ResolveTypedef( const std::string& name ); } }

// Check the result of this:
this_dll.G__defined_tagname("vector<TH1*,allocator<TH1*> >", 4)
"""

import inspect
import os
import sys
import types

from os.path import basename, dirname, exists, join as pjoin

import ROOT

import rootpy.userdata as userdata

from .. import log; log = log[__name__]
from rootpy.defaults import extra_initialization
from rootpy.extern.module_facade import Facade

def mtime(path):
    return os.stat(path).st_mtime

MODULES_PATH = None

@extra_initialization
def initialize():
    global MODULES_PATH
    MODULES_PATH = pjoin(userdata.BINARY_PATH, 'modules')
    if not exists(MODULES_PATH):
        os.makedirs(MODULES_PATH)
    
    # Used insetad of AddDynamicPath for ordering
    path = ":".join([MODULES_PATH, ROOT.gSystem.GetDynamicPath()])
    ROOT.gSystem.SetDynamicPath(path)

class Namespace(object):
    """
    Represents a sub-namespace
    """
        
class FileCode(object):
    def __init__(self, filename, callermodule):
        self.filename = filename
        self.module = callermodule
        self.name = self.module + "." + basename(self.filename)
        self.loaded = False
    
    @property
    def mtime(self):
        return mtime(self.filename)
    
    @property
    def compiled_path(self):
        ext = "." + ROOT.gSystem.GetSoExt()
        return pjoin(MODULES_PATH, self.name + ext)
    
    @property
    def compiled(self):
        return exists(self.compiled_path) and mtime(self.compiled_path) > self.mtime
        
    def load(self):
        
        if not self.compiled:
            log.info("Compiling {0}".format(self.compiled_path))
            ROOT.gSystem.CompileMacro(self.filename, 'k-', self.name, MODULES_PATH)
        else:
            log.debug("Loading existing {0}".format(self.compiled_path))
            ROOT.gInterpreter.Load(self.compiled_path)
        self.loaded = True

    def get(self, name):
        if not self.loaded:
            self.load()
            
        return getattr(ROOT, name)

@Facade(__name__, expose_internal=False)
class Compiled(object):
    
    registered_code = {}
    debug = False
    optimize = True
    
    @property
    def caller_location(self):
        caller = sys._getframe(2)
        caller_file = inspect.getfile(caller)
        caller_module = inspect.getmodule(caller)
        caller_directory = dirname(caller_file)
        
        return caller_directory, caller_module.__name__, caller.f_lineno
    
    def get_symbol(self, symbol):
        if symbol in self.registered_code:
            return self.registered_code[symbol].get(symbol)
    
    def __getattr__(self, what):
        return self.get_symbol(what)
    
    def register_code(self, code, symbols):
        raise NotImplementedError
    
    def register_file(self, filename, symbols):
        caller_directory, caller_modulename, _ = self.caller_location
    
        absfile = pjoin(caller_directory, filename)
    
        if not exists(absfile):
            raise RuntimeError("Can't find file {0}".format(absfile))
        
        code = FileCode(absfile, caller_modulename)
        for s in symbols:
            self.registered_code[s] = code

