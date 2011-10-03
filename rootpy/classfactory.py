"""
This module handles dictionary generation of classes for use
in the Python environment. Dictionaried are kept in
$ROOTPY_DATA for later use so they are not repeatedly regenerated
"""
import ROOT
import os
import re
import atexit
import uuid
from rootpy.userdata import DATA_ROOT

__ROOT_version = str(ROOT.gROOT.GetVersionCode())
__loaded_dicts = {}
__dicts_path = os.path.join(DATA_ROOT, 'dicts', __ROOT_version)
ROOT.gSystem.SetDynamicPath(":".join([__dicts_path, ROOT.gSystem.GetDynamicPath()]))
__lookup_table_name = 'lookup'

if not os.path.exists(__dicts_path):
    os.makedirs(__dicts_path)

if os.path.exists(os.path.join(__dicts_path, __lookup_table_name)):
    __lookup_file = open(os.path.join(__dicts_path, __lookup_table_name), 'r')
    __lookup_table = dict([reversed(line.strip().split('\t')) for line in __lookup_file.readlines()])
    __lookup_file.close()
else:
    __lookup_table = {}

def generate(declaration, headers = None):
    
    if headers is not None:
        headers = sorted(headers.split(';'))
        unique_name = ';'.join([declaration]+headers)
    else:
        unique_name = declaration

    # If this class was previously requested, do nothing
    if __loaded_dicts.has_key(unique_name):
        return True
    
    # If as .so already exists for this class, use it.
    if __lookup_table.has_key(unique_name):
        if ROOT.gSystem.Load("%s.so"% __lookup_table[unique_name]) in (0, 1):
            __loaded_dicts[unique_name] = None
            return True
        return False
    
    # This dict was not previously generated so we must create it now
    print "generating dictionary for %s..." % declaration
    source = ""
    if headers is not None:
        for header in headers:
            if re.match('^<.+>$', header):
                source += "#include %s\n"% header
            else:
                source += "#include \"%s\"\n"% header
    source += "#ifdef __CINT__\n"
    source += "#pragma link C++ class %s+;\n"% declaration
    source += "#else\n"
    source += "using namespace std;\n"
    source += "template class %s;\n"% declaration
    source += "#endif\n"
    
    dict_id = uuid.uuid4().hex
    sourcefilename = os.path.join(__dicts_path, "%s.C"% dict_id)
    sourcefile = open(sourcefilename,'w')
    sourcefile.write(source)
    sourcefile.close()
    msg_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    success = ROOT.gSystem.CompileMacro(sourcefilename, 'k-', dict_id, __dicts_path) == 1
    ROOT.gErrorIgnoreLevel = msg_level
    if success:
        __lookup_table[unique_name] = dict_id
        __loaded_dicts[unique_name] = None
    else:
        os.unlink(sourcefilename)
        try:
            os.unlink(os.path.join(__dicts_path, "%s_C.d"% dict_id))
            os.unlink(os.path.join(__dicts_path, "%s.so"% dict_id))
        except: pass
    return success

@atexit.register
def __cleanup():
    __lookup_file = open(os.path.join(__dicts_path, __lookup_table_name), 'w')
    for name, dict_id in __lookup_table.items():
        __lookup_file.write("%s\t%s\n"% (dict_id, name))
    __lookup_file.close()
