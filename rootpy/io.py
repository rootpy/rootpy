import ROOT
from rootpy.utils import asrootpy

class File(ROOT.TFile):

    def __init__(self, *args, **kwargs):

        ROOT.TFile.__init__(self, *args)

    def Get(self, name):

        return asrootpy(ROOT.TFile.Get(name))
