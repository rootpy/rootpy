import string
import ROOT
import os
import re
import atexit

__classes = {}

__dicts_path = os.path.join(os.environ['ROOTPY_CONFIG'], 'dicts')
if not os.path.exists(__dicts_path):
    os.mkdir(__dicts_path)

__lookup_file = open(os.path.join(__dicts_path, 'lookup_table'))
__lookup_table = dict([line.split() for line in __lookup_file.readlines()])

def make_class(declaration, headers=None):
    
    if __classes.has_key(declaration):
        return True
    source = ""
    if headers is not None:
        headers = headers.split(';')
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
    
    tmpfilename = os.path.join(__dicts_path, "%s.C"% uuid.uuid4().hex)
    tmpfile = open(tmpfilename,'w')
    tmpfile.write(source)
    tmpfile.close()
    msg_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    success = ROOT.gROOT.ProcessLine(".L %s+"% tmpfilename) == 0
    ROOT.gErrorIgnoreLevel = msg_level
    if success:
        __classes[declaration] = tmpfilename
    else:
        os.unlink(tmpfilename)
        os.unlink("%s.d"% tmpfilename.replace('.','_'))
        os.unlink("%s.so"% tmpfilename.replace('.','_'))
    return success

@atexit.register
def __cleanup():
   __lookup_file.close() 
