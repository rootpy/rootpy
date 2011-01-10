"""
This module enhances IO-related ROOT funcionality
"""
import ROOT
from rootpy.utils import asrootpy

class File(ROOT.TFile):
    """
    Inherits from TFile
    """
    def __init__(self, *args, **kwargs):

        ROOT.TFile.__init__(self, *args)

    def Get(self, name):
        """
        Attempt to convert requested object into rootpy form
        """
        return asrootpy(ROOT.TFile.Get(self, name))
