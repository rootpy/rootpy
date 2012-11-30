import hashlib
import inspect
import os
import pkg_resources
import sys
import textwrap
import types

from os.path import basename, dirname, exists, join as pjoin

import ROOT

import rootpy.userdata as userdata

from .. import log; log = log[__name__]
from rootpy.defaults import extra_initialization
from rootpy.extern.module_facade import Facade

def mtime(path):
    return os.stat(path).st_mtime

MODULES_PATH = pjoin(userdata.BINARY_PATH, 'modules')
if not exists(MODULES_PATH):
    os.makedirs(MODULES_PATH)

@extra_initialization
def initialize():
    # Used instead of AddDynamicPath for ordering
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
    
    def caller_location(self, depth=0):
        caller = sys._getframe(depth+2)
        caller_file = inspect.getfile(caller)
        caller_module = inspect.getmodule(caller)
        if caller_module:
            caller_module = caller_module.__name__
            # Note: caller_file may be a relative path from $PWD at python startup
            #       therefore, to get a solid abspath:
            caller_directory = pkg_resources.get_provider(caller_module).module_path
        else:
            caller_module = "..unknown.."
            caller_directory = dirname(caller_file)
        
        return caller_directory, caller_module, caller.f_lineno
    
    def get_symbol(self, symbol):
        if symbol in self.registered_code:
            return self.registered_code[symbol].get(symbol)
    
    def __getattr__(self, what):
        return self.get_symbol(what)
    
    def register_code(self, code, symbols):
        """
        Note: This assumes that call site occurs exactly once.
              If you don't do that, you're better off writing to a temporary
              file and calling `register_file`
        """
        filename = hashlib.sha1(code).hexdigest()[:8] + ".cxx"
        filepath = pjoin(MODULES_PATH, filename)
        
        _, caller_modulename, lineno = self.caller_location()
        
        #code += "#line {0} {1}".format(caller_modulename, lineno)
        if not exists(filepath):
            # Only write it if it doesn't exist (1/4billion chance of collision)
            with open(filepath, "w") as fd:
                fd.write(textwrap.dedent(code))
            
        code = FileCode(filepath, caller_modulename)
        self.register(code, symbols)
        
    def register(self, code, symbols):
        for s in symbols:
            self.registered_code[s] = code
    
    def register_file(self, filename, symbols):
        caller_directory, caller_modulename, _ = self.caller_location()
    
        absfile = pjoin(caller_directory, filename)
    
        if not exists(absfile):
            raise RuntimeError("Can't find file {0}".format(absfile))
        
        code = FileCode(absfile, caller_modulename)
        self.register(code, symbols)

