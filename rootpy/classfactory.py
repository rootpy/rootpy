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
from rootpy.userdata import DATA_ROOT


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
    if unique_name in __LOOKUP_TABLE:
        if ROOT.gSystem.Load('%s.so' % __LOOKUP_TABLE[unique_name]) in (0, 1):
            __LOADED_DICTS[unique_name] = None
            return True
        return False

    # This dict was not previously generated so we must create it now
    print "generating dictionary for %s..." % declaration
    source = ""
    if headers is not None:
        for header in headers:
            if re.match('^<.+>$', header):
                source += '#include %s\n' % header
            else:
                source += '#include "%s"\n' % header
    source += '#ifdef __CINT__\n'
    source += '#pragma link C++ class %s+;\n' % declaration
    source += '#else\n'
    source += 'using namespace std;\n'
    source += 'template class %s;\n' % declaration
    source += '#endif\n'

    dict_id = uuid.uuid4().hex
    sourcefilename = os.path.join(__DICTS_PATH, '%s.C' % dict_id)
    sourcefile = open(sourcefilename, 'w')
    sourcefile.write(source)
    sourcefile.close()
    msg_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    # TODO use cpp compiler
    success = ROOT.gSystem.CompileMacro(sourcefilename, 'k-', dict_id, __DICTS_PATH) == 1
    ROOT.gErrorIgnoreLevel = msg_level
    if success:
        __LOOKUP_TABLE[unique_name] = dict_id
        __LOADED_DICTS[unique_name] = None
        __NEW_DICTS = True
    else:
        os.unlink(sourcefilename)
        try:
            os.unlink(os.path.join(__DICTS_PATH, '%s_C.d' % dict_id))
            os.unlink(os.path.join(__DICTS_PATH, '%s.so' % dict_id))
        except:
            pass
    return success


@atexit.register
def __cleanup():

    if __NEW_DICTS:
        file = open(os.path.join(__DICTS_PATH, __LOOKUP_TABLE_NAME), 'w')
        for name, dict_id in __LOOKUP_TABLE.items():
            file.write('%s\t%s\n' % (dict_id, name))
        file.close()
