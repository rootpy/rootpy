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
from rootpy.userdata import DATA_ROOT


LINKDEF = '''\
%(includes)s
#ifdef __CINT__
#pragma link off all global;
#pragma link off all class;
#pragma link off all function;
#pragma link off all typedef;
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
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
if sys.maxsize > 2 ** 32:
    NBITS = '64'
else:
    NBITS = '32'
ROOT_VERSION = str(ROOT.gROOT.GetVersionCode())
LOADED_DICTS = {}
DICTS_PATH = os.path.join(DATA_ROOT, 'dicts', NBITS, ROOT_VERSION)
ROOT.gSystem.SetDynamicPath(":".join([DICTS_PATH, ROOT.gSystem.GetDynamicPath()]))
LOOKUP_TABLE_NAME = 'lookup'

if not os.path.exists(DICTS_PATH):
    os.makedirs(DICTS_PATH)

if os.path.exists(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME)):
    LOOKUP_FILE = open(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME), 'r')
    LOOKUP_TABLE = dict([reversed(line.strip().split('\t')) for line in LOOKUP_FILE.readlines()])
    LOOKUP_FILE.close()
else:
    LOOKUP_TABLE = {}


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
        return
    # If as .so already exists for this class, use it.
    if unique_name in LOOKUP_TABLE:
        if ROOT.gSystem.Load('%s.so' % LOOKUP_TABLE[unique_name]) != 0:
            raise RuntimeError("Failed to load the library for '%s'" %
                    declaration)
        LOADED_DICTS[unique_name] = None
        return
    # This dict was not previously generated so we must create it now
    if verbose:
        print "generating dictionary for %s..." % declaration
    includes = ''
    if headers is not None:
        for header in headers:
            if re.match('^<.+>$', header):
                includes += '#include %s\n' % header
            else:
                includes += '#include "%s"\n' % header
    source = LINKDEF % locals()
    dict_id = uuid.uuid4().hex
    sourcepath = os.path.join(DICTS_PATH, 'LinkDef.h')
    with open(sourcepath, 'w') as sourcefile:
        sourcefile.write(source)
    cwd = os.getcwd()
    os.chdir(DICTS_PATH)
    # call rootcint to generate the dictionaries
    rootcint_cmd = (
            "rootcint -f dict.cxx "
            "-c -p -I{ROOT_INC} LinkDef.h").format(
                    **dict(globals(), **locals()))
    if verbose:
        print rootcint_cmd
    if subprocess.call(rootcint_cmd, shell=True) != 0:
        os.chdir(cwd)
        raise RuntimeError("rootcint failed for '%s'" % declaration)
    # rootcint forgets to put the includes in the generated cxx
    # manually add them here
    with open('dict_patched.cxx', 'w') as patched_source:
        patched_source.write(includes)
        with open('dict.cxx', 'r') as orig_source:
            orig_lines = orig_source.read()
        patched_source.write(orig_lines)
    os.rename('dict_patched.cxx', 'dict.cxx')
    # compile the dictionaries
    compile_cmd = (
            '{CXX} -fPIC {ROOT_CXXFLAGS} -I. -Wall -c dict.cxx -o dict.o'
            ).format(**dict(globals(), **locals()))
    if verbose:
        print compile_cmd
    if subprocess.call(compile_cmd, shell=True) != 0:
        os.chdir(cwd)
        raise RuntimeError("failed to compile '%s'" % declaration)
    # create a shared library
    link_cmd = (
            '{LD} {ROOT_LDFLAGS} -shared -Wall dict.o -o {dict_id}.so'
            ).format(**dict(globals(), **locals()))
    if verbose:
        print link_cmd
    if subprocess.call(link_cmd, shell=True) != 0:
        os.chdir(cwd)
        raise RuntimeError("Failed to created library for '%s'" % declaration)
    # load the library
    if ROOT.gSystem.Load('%s.so' % dict_id) != 0:
        os.chdir(cwd)
        raise RuntimeError("Failed to load the library for '%s'" % declaration)
    # clean up
    os.unlink('LinkDef.h')
    os.unlink('dict.cxx')
    os.unlink('dict.h')
    os.unlink('dict.o')
    os.chdir(cwd)
    LOOKUP_TABLE[unique_name] = dict_id
    LOADED_DICTS[unique_name] = None
    NEW_DICTS = True


@atexit.register
def cleanup():
    if NEW_DICTS:
        file = open(os.path.join(DICTS_PATH, LOOKUP_TABLE_NAME), 'w')
        for name, dict_id in LOOKUP_TABLE.items():
            file.write('%s\t%s\n' % (dict_id, name))
        file.close()
