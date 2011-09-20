"""
This module enhances IO-related ROOT funcionality
"""
import ROOT
from ..core import camelCaseMethods
from ..register import register
from ..utils import asrootpy
from . import utils


@camelCaseMethods
@register
class Directory(ROOT.TDirectory):
    """
    Inherits from TDirectory
    """

    def __init__(self, *args, **kwargs):

        ROOT.TDirectory.__init__(self, *args)

    def Get(self, name):
        """
        Attempt to convert requested object into rootpy form
        """
        return asrootpy(ROOT.TDirectory.Get(self, name))

    def GetDirectory(self, name):
        """
        Should return a Directory object rather than TDirectory
        """
        #TODO: how to get asrootpy to return a Directory object?
        return asrootpy(ROOT.TDirectory.GetDirectory(self, name))

    def walk(self, top=None):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top)


@camelCaseMethods
class File(Directory, ROOT.TFile):
    """
    Inherits from Directory
    """
    
    def __init__(self, *args, **kwargs):

        ROOT.TFile.__init__(self, *args)


def open(filename, mode=""):

    file = ROOT.TFile.Open(filename, mode)
    if not file:
        raise IOError("No such file: '%s'"% filename)
    file.__class__ = File
    return file
