"""
This module enhances IO-related ROOT funcionality
"""
import ROOT
from ..core import camelCaseMethods
from ..registry import register
from ..utils import asrootpy
from . import utils

class DoesNotExist(Exception):
    pass


class _DirectoryBase(object):
    """
    A mixin (can't stand alone). To be improved.
    """
    
    def walk(self, top=None):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top)

    def __getattr__(self, attr):
        """
        Natural naming support.
        Now you can get an object from a File/Directory with
        myfile.somedir.otherdir.histname
        """
        return self.Get(attr)
            
    def Get(self, name):
        """
        Attempt to convert requested object into rootpy form
        """
        thing = asrootpy(self.__class__.__bases__[-1].Get(self, name))
        if not thing:
            raise DoesNotExist("requested path/object '%s' does not exist in %s" % (name, self._path))
        if isinstance(thing, _DirectoryBase):
            thing._path = '/'.join([self._path, name])
        return thing

    def GetDirectory(self, name):
        """
        Should return a Directory object rather than TDirectory
        """
        #TODO: how to get asrootpy to return a Directory object?
        return asrootpy(self.__class__.__bases__[-1].GetDirectory(self, name))


@camelCaseMethods
@register
class Directory(_DirectoryBase, ROOT.TDirectoryFile):
    """
    Inherits from TDirectory
    """

    def __init__(self, name, *args, **kwargs):

        self._path = name
        ROOT.TDirectoryFile.__init__(self, name, *args)


@camelCaseMethods
class File(_DirectoryBase, ROOT.TFile):
    """
    Inherits from Directory
    """
    
    def __init__(self, name, *args, **kwargs):

        self._path = name
        ROOT.TFile.__init__(self, name, *args)


def open(filename, mode=""):

    file = ROOT.TFile.Open(filename, mode)
    if not file:
        raise IOError("No such file: '%s'"% filename)
    file.__class__ = File
    file._path = filename
    return file
