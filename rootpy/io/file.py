"""
This module enhances IO-related ROOT funcionality
"""
import ROOT
from ..core import camelCaseMethods
from ..utils import asrootpy

@camelCaseMethods
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

def open(filename, mode=""):

    file = ROOT.TFile.Open(filename, mode)
    if not file:
        raise IOError("No such file: '%s'"% filename)
    file.__class__ = File
    return file
