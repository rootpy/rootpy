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
import rootpy.userdata as userdata

LINKDEF = '''\
%(includes)s
#ifdef __CINT__
#pragma link C++ class %(declaration)s+;
#else
using namespace std;
template class %(declaration)s;
#endif
'''


def root_config(*flags):

    flags = subprocess.Popen(
        ['root-config'] + list(flags),
        stdout=subprocess.PIPE).communicate()[0].strip().split()
    flags = ' '.join(['-I'+os.path.realpath(p[2:]) if
        p.startswith('-I') else p for p in flags])
    return flags


ROOT_INC = root_config('--incdir')
ROOT_LDFLAGS = root_config('--libs', '--ldflags')
ROOT_CXXFLAGS = root_config('--cflags')
CXX = root_config('--cxx')
LD = root_config('--ld')

NEW_DICTS = False
LOOKUP_TABLE_NAME = 'lookup'

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

def generate(declaration, headers=None, verbose=False):

    global NEW_DICTS

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
        cwd = os.getcwd()
        os.chdir(DICTS_PATH)
        if ROOT.gSystem.Load('%s.so' % LOOKUP_TABLE[unique_name]) not in (0, 1):
            os.chdir(cwd)
            raise RuntimeError("Failed to load the library for '%s'" %
                    declaration)
        os.chdir(cwd)
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
    sourcepath = os.path.join(DICTS_PATH, '%s.C' % dict_id)
    with open(sourcepath, 'w') as sourcefile:
        sourcefile.write(source)
    log.debug("Source path: {0}".format(sourcepath))
    if ROOT.gSystem.CompileMacro(sourcepath, 'k-', dict_id, DICTS_PATH) != 1:
        raise RuntimeError("Failed to load the library for '%s'" % declaration)
    LOOKUP_TABLE[unique_name] = dict_id
    LOADED_DICTS[unique_name] = None
    NEW_DICTS = True


@atexit.register
def cleanup():
    if NEW_DICTS:
        with open(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME), 'w') as dfile:
            for name, dict_id in LOOKUP_TABLE.items():
                dfile.write('%s\t%s\n' % (dict_id, name))
