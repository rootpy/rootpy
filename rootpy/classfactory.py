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

ROOT_INC = subprocess.Popen(
    ['root-config', '--incdir'],
    stdout=subprocess.PIPE).communicate()[0].strip().split()
ROOT_INC = ' '.join([os.path.realpath(p) for p in ROOT_INC])
ROOT_LDFLAGS = subprocess.Popen(
    ['root-config', '--libs', '--ldflags'],
    stdout=subprocess.PIPE).communicate()[0].strip()
ROOT_CFLAGS = subprocess.Popen(
    ['root-config', '--cflags'],
    stdout=subprocess.PIPE).communicate()[0].strip()


__NEW_DICTS = False
if sys.maxsize > 2 ** 32:
    __NBITS = '64'
else:
    __NBITS = '32'
__ROOT_VERSION = str(ROOT.gROOT.GetVersionCode())
__LOADED_DICTS = {}
__DICTS_PATH = os.path.join(DATA_ROOT, 'dicts', __NBITS, __ROOT_VERSION)
ROOT.gSystem.SetDynamicPath(":".join([__DICTS_PATH, ROOT.gSystem.GetDynamicPath()]))
__LOOKUP_TABLE_NAME = 'lookup'

if not os.path.exists(__DICTS_PATH):
    os.makedirs(__DICTS_PATH)

if os.path.exists(os.path.join(__DICTS_PATH, __LOOKUP_TABLE_NAME)):
    __LOOKUP_FILE = open(os.path.join(__DICTS_PATH, __LOOKUP_TABLE_NAME), 'r')
    __LOOKUP_TABLE = dict([reversed(line.strip().split('\t')) for line in __LOOKUP_FILE.readlines()])
    __LOOKUP_FILE.close()
else:
    __LOOKUP_TABLE = {}


def generate(declaration, headers=None):

    global __NEW_DICTS

    if headers:
        if isinstance(headers, basestring):
            headers = sorted(headers.split(';'))
        unique_name = ';'.join([declaration] + headers)
    else:
        unique_name = declaration
    unique_name = unique_name.replace(' ', '')
    # The library is already loaded, do nothing
    if unique_name in __LOADED_DICTS:
        return True

    # If as .so already exists for this class, use it.
    """
    if unique_name in __LOOKUP_TABLE:
        if ROOT.gSystem.Load('%s.so' % __LOOKUP_TABLE[unique_name]) in (0, 1):
            __LOADED_DICTS[unique_name] = None
            return True
        return False
    """
    # This dict was not previously generated so we must create it now
    print "generating dictionary for %s..." % declaration
    includes = ''
    if headers is not None:
        for header in headers:
            if re.match('^<.+>$', header):
                includes += '#include %s\n' % header
            else:
                includes += '#include "%s"\n' % header
    source = LINKDEF % locals()
    print source
    dict_id = uuid.uuid4().hex
    sourcepath = os.path.join(__DICTS_PATH, 'LinkDef.h')
    with open(sourcepath, 'w') as sourcefile:
        sourcefile.write(source)
    rootcint = (
            "rootcint -f {dict_id}.cxx "
            "-c -p -I{ROOT_INC} LinkDef.h").format(
                    **dict(globals(), **locals()))
    print rootcint
    cwd = os.getcwd()
    os.chdir(__DICTS_PATH)
    subprocess.call(rootcint, shell=True)
    os.chdir(cwd)
    __LOOKUP_TABLE[unique_name] = dict_id
    __LOADED_DICTS[unique_name] = None
    __NEW_DICTS = True
    #os.unlink(sourcefilename)
    return True


@atexit.register
def __cleanup():

    if __NEW_DICTS:
        file = open(os.path.join(__DICTS_PATH, __LOOKUP_TABLE_NAME), 'w')
        for name, dict_id in __LOOKUP_TABLE.items():
            file.write('%s\t%s\n' % (dict_id, name))
        file.close()
