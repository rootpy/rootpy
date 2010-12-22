import ROOT
import uuid
import os

__classes = {}

def make_class(declaration, headers=None):
    
    if __classes.has_key(declaration):
        return True
    source = ""
    if headers:
        if type(headers) is not list:
            headers = [headers]
        for header in headers:
            source += "#include %s\n"% header
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
    success = ROOT.gROOT.ProcessLine(".L %s+"% tmpfilename) == 0
    #os.unlink(tmpfilename)
    #os.unlink("%s.d"% tmpfilename.replace('.','_'))
    #os.unlink("%s.so"% tmpfilename.replace('.','_'))
    if success:
        __classes[declaration] = None
    return success
