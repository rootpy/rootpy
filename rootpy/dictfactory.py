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

__loaded_dicts = {}

__dicts_path = os.path.join(DATA_ROOT, 'dicts')
if not os.path.exists(__dicts_path):
    os.mkdir(__dicts_path)

if os.path.exists(os.path.join(__dicts_path, 'lookup_table')):
    __lookup_file = open(os.path.join(__dicts_path, 'lookup_table'), 'r')
    __lookup_table = dict([reversed(line.strip().split('\t')) for line in __lookup_file.readlines()])
    __lookup_file.close()
else:
    __lookup_table = {}
print __lookup_table

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
        if ROOT.gSystem.Load(os.path.join(__dicts_path, __lookup_table[unique_name]+".so")) == 0:
            __loaded_dicts[unique_name] = None
            return True
        return False
    
    # This dict was not previously generated so we must create it now
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
    success = ROOT.gROOT.ProcessLine(".L %s+"% sourcefilename) == 0
    ROOT.gErrorIgnoreLevel = msg_level
    if success:
        __lookup_table[unique_name] = dict_id
        __loaded_dicts[unique_name] = None
    else:
        os.unlink(sourcefilename)
        os.unlink("%s.d"% sourcefilename.replace('.','_'))
        os.unlink("%s.so"% sourcefilename.replace('.','_'))
    return success

@atexit.register
def __cleanup():
    __lookup_file = open(os.path.join(__dicts_path, 'lookup_table'), 'w')
    for name, dict_id in __lookup_table.items():
        __lookup_file.write("%s\t%s\n"% (dict_id, name))
    __lookup_file.close()
