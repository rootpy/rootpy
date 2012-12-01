"""
This module handles dictionary generation of classes for use
in the Python environment. Dictionaries are kept in
$ROOTPY_DATA for later use so they are not repeatedly regenerated
"""
import ROOT
import os
import sys
import re
import atexit
import uuid
import subprocess

from . import log; log = log[__name__]
from rootpy.defaults import extra_initialization

import rootpy.compiled as compiled
import rootpy.userdata as userdata

compiled.register_code("""
    #include <string>
    
    // PyROOT builtin
    namespace PyROOT { namespace Utility {
        const std::string ResolveTypedef( const std::string& name );
    } }
    
    // cint magic
    int G__defined_tagname(const char*, int);
    
    // Returns true if the given type does not require a dictionary
    bool _rootpy_dictionary_already_exists(const char* type) {
        const std::string full_typedef = PyROOT::Utility::ResolveTypedef(type);
        return G__defined_tagname(full_typedef.c_str(), 4) != -1;
    }
""", ["_rootpy_dictionary_already_exists"])

LINKDEF = '''\
%(includes)s
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;
#pragma link C++ class %(declaration)s;
#pragma link C++ class %(declaration)s::*;
#ifdef HAS_ITERATOR
#pragma link C++ operators %(declaration)s::iterator;
#pragma link C++ operators %(declaration)s::const_iterator;
#pragma link C++ operators %(declaration)s::reverse_iterator;
#pragma link C++ operators %(declaration)s::const_reverse_iterator;
#endif
#endif
'''

def root_config(*flags):

    flags = subprocess.Popen(
        ['root-config'] + list(flags),
        stdout=subprocess.PIPE).communicate()[0].strip().split()
    flags = ' '.join(['-I'+os.path.realpath(p[2:]) if
        p.startswith('-I') else p for p in flags])
    return flags


def shell(cmd):

    log.debug(cmd)
    return subprocess.call(cmd, shell=True)


ROOT_INC = root_config('--incdir')
ROOT_LDFLAGS = root_config('--libs', '--ldflags')
ROOT_CXXFLAGS = root_config('--cflags')
CXX = root_config('--cxx')
LD = root_config('--ld')

NEW_DICTS = False
LOOKUP_TABLE_NAME = 'lookup'
USE_ACLIC = True

# Initialized in initialize()
LOOKUP_TABLE = {}
LOADED_DICTS = {}
DICTS_PATH = None


@extra_initialization
def initialize():
    global LOOKUP_TABLE, DICTS_PATH

    DICTS_PATH = os.path.join(userdata.BINARY_PATH, 'dicts')

    # Used insetad of AddDynamicPath for ordering
    path = ":".join([DICTS_PATH, ROOT.gSystem.GetDynamicPath()])
    ROOT.gSystem.SetDynamicPath(path)

    if not os.path.exists(DICTS_PATH):
        os.makedirs(DICTS_PATH)

    if os.path.exists(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME)):
        LOOKUP_FILE = open(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME), 'r')
        LOOKUP_TABLE = dict([reversed(line.strip().split('\t'))
                             for line in LOOKUP_FILE.readlines()])
        LOOKUP_FILE.close()


def generate(declaration,
        headers=None, has_iterators=False):

    global NEW_DICTS
    
    if compiled._rootpy_dictionary_already_exists(declaration):
        log.debug("generate({0}) => already available".format(declaration))
        return

    log.debug("requesting dictionary for %s" % declaration)
    if headers:
        if isinstance(headers, basestring):
            headers = sorted(headers.split(';'))
        unique_name = ';'.join([declaration] + headers)
    else:
        unique_name = declaration
    unique_name = unique_name.replace(' ', '')
    # The library is already loaded, do nothing
    if unique_name in LOADED_DICTS:
        log.debug("dictionary for {0} is already loaded".format(declaration))
        return
    # If as .so already exists for this class, use it.
    if unique_name in LOOKUP_TABLE:
        log.debug("loading previously generated dictionary for {0}"
                  .format(declaration))
        if ROOT.gInterpreter.Load(
                os.path.join(DICTS_PATH, '%s.so' % LOOKUP_TABLE[unique_name])) not in (0, 1):
            raise RuntimeError("failed to load the library for '%s'" %
                    declaration)
        LOADED_DICTS[unique_name] = None
        return

    # This dict was not previously generated so we must create it now
    log.info("generating dictionary for {0} ...".format(declaration))
    includes = ''
    if headers is not None:
        for header in headers:
            if re.match('^<.+>$', header):
                includes += '#include %s\n' % header
            else:
                includes += '#include "%s"\n' % header
    source = LINKDEF % locals()
    dict_id = uuid.uuid4().hex
    if USE_ACLIC:
        sourcepath = os.path.join(DICTS_PATH, '%s.C' % dict_id)
        log.debug("source path: {0}".format(sourcepath))
        with open(sourcepath, 'w') as sourcefile:
            sourcefile.write(source)
        if ROOT.gSystem.CompileMacro(sourcepath, 'k-', dict_id, DICTS_PATH) != 1:
            raise RuntimeError("failed to load the library for '%s'" % declaration)
    else:
        cwd = os.getcwd()
        os.chdir(DICTS_PATH)
        sourcepath = os.path.join(DICTS_PATH, 'LinkDef.h')
        OPTS_FLAGS = ''
        if has_iterators:
            OPTS_FLAGS = '-DHAS_ITERATOR'
        all_vars = dict(globals(), **locals())
        with open(sourcepath, 'w') as sourcefile:
            sourcefile.write(source)
        # run rootcint
        if shell(('rootcint -f dict.cxx -c -p %(OPTS_FLAGS)s '
                  '-I%(ROOT_INC)s LinkDef.h') % all_vars):
            os.chdir(cwd)
            raise RuntimeError('rootcint failed for %s' % declaration)
        # add missing includes
        os.rename('dict.cxx', 'dict.tmp')
        with open('dict.cxx', 'w') as patched_source:
            patched_source.write(includes)
            with open('dict.tmp', 'r') as orig_source:
                patched_source.write(orig_source.read())
        if shell(('%(CXX)s %(ROOT_CXXFLAGS)s %(OPTS_FLAGS)s '
                  '-Wall -fPIC -c dict.cxx -o dict.o') %
                  all_vars):
            os.chdir(cwd)
            raise RuntimeError('failed to compile %s' % declaration)
        if shell(('%(LD)s %(ROOT_LDFLAGS)s -Wall -shared '
               'dict.o -o %(dict_id)s.so') % all_vars):
            os.chdir(cwd)
            raise RuntimeError('failed to link %s' % declaration)
        # load the newly compiled library
        if ROOT.gInterpreter.Load('%s.so' % dict_id) not in (0, 1):
            os.chdir(cwd)
            raise RuntimeError('failed to load the library for %s' % declaration)
        os.chdir(cwd)

    LOOKUP_TABLE[unique_name] = dict_id
    LOADED_DICTS[unique_name] = None
    NEW_DICTS = True


@atexit.register
def cleanup():
    if NEW_DICTS:
        with open(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME), 'w') as dfile:
            for name, dict_id in LOOKUP_TABLE.items():
                dfile.write('%s\t%s\n' % (dict_id, name))
