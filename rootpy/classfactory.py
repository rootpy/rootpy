import ROOT
import uuid
import os

def make_class(declaration, headers=None):

    source = ""
    if headers:
        if type(headers) is not list:
            headers = [headers]
        for header in headers:
            source += "#include %s\n"% header
    source += "#ifdef __CINT__\n"
    source += "#pragma link C++ class %s;\n"% declaration
    source += "#else\n"
    source += "using namespace std;\n"
    source += "template class %s;\n"% declaration
    source += "#endif\n"
    tmpfilename = "%s.h"% uuid.uuid4().hex
    tmpfile = open(tmpfilename,'w')
    tmpfile.write(source)
    level = ROOT.gErrorIgnoreLevel
    #ROOT.gErrorIgnoreLevel = ROOT.kFatal
    ROOT.gROOT.ProcessLine(".L %s+"% tmpfilename)
    print ROOT.vector("vector<int>")
    #os.unlink(tmpfilename)
    os.unlink("%s.d"% tmpfilename.replace('.','_'))
    os.unlink("%s.so"% tmpfilename.replace('.','_'))
    ROOT.gErrorIgnoreLevel = level
