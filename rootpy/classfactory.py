import ROOT
import uuid
import os
import atexit
import re

__classes = {}

def make_class(declaration, headers=None):
    
    if __classes.has_key(declaration):
        return True
    source = ""
    if headers:
        if type(headers) is not list:
            headers = [headers]
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
    tmpfilename = "%s.C"% uuid.uuid4().hex
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
    
    for tmpfilename in __classes.values():
        os.unlink(tmpfilename)
        os.unlink("%s.d"% tmpfilename.replace('.','_'))
        os.unlink("%s.so"% tmpfilename.replace('.','_'))
