"""
This module enhances IO-related ROOT functionality
"""
import ROOT
from ..core import camelCaseMethods
from ..registry import register
from ..utils import asrootpy
from . import utils
from .. import path
from contextlib import contextmanager
import os

VALIDPATH = '^(?P<file>.+.root)(?:[/](?P<path>.+))?$'


class DoesNotExist(Exception):
    pass


class _DirectoryBase(object):
    """
    A mixin (can't stand alone). To be improved.
    """
    
    def walk(self, top=None, pattern=None):
        """
        Calls :func:`rootpy.io.utils.walk`.
        """
        return utils.walk(self, top, pattern)

    def __getattr__(self, attr):
        """
        Natural naming support.
        Now you can get an object from a File/Directory with
        myfile.somedir.otherdir.histname
        """
        return self.Get(attr)
            
    
    def __getitem__(self, name):

        return self.Get(name)
    

    def Get(self, name):
        """
        Attempt to convert requested object into rootpy form
        """
        thing = asrootpy(self.__class__.__bases__[-1].Get(self, name))
        if not thing:
            raise DoesNotExist("requested path/object '%s' does not exist in %s" % (name, self._path))
        if isinstance(thing, _DirectoryBase):
            if name != '.':
                if isinstance(thing, File):
                    thing._path = os.path.normpath((':' + os.path.sep).join([self._path, name]))
                else:
                    thing._path = os.path.normpath(os.path.join(self._path, name))
        return thing

    def GetDirectory(self, name):
        """
        Should return a Directory object rather than TDirectory
        """
        #TODO: how to get asrootpy to return a Directory object?
        dir = asrootpy(self.__class__.__bases__[-1].GetDirectory(self, name))
        if not dir:
            raise DoesNotExist("requested path '%s' does not exist in %s" % (name, self._path))
        if name != '.':
            if isinstance(dir, File):
                dir._path = os.path.normpath((':' + os.path.sep).join([self._path, name]))
            else:
                dir._path = os.path.normpath(os.path.join(self._path, name))
        return dir

    
@camelCaseMethods
@register()
class Directory(_DirectoryBase, ROOT.TDirectoryFile):
    """
    Inherits from TDirectory
    """

    def __init__(self, name, *args, **kwargs):

        self._path = name
        ROOT.TDirectoryFile.__init__(self, name, *args)
    
    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()


@camelCaseMethods
class File(_DirectoryBase, ROOT.TFile):
    """
    Inherits from Directory
    """
    
    def __init__(self, *args, **kwargs):

        ROOT.TFile.__init__(self, *args, **kwargs)
        self._path = self.GetName()
    
    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):
        
        self.Close()
        return False 
    
    def __str__(self):

        return "%s('%s')" % (self.__class__.__name__, self._path)

    def __repr__(self):

        return self.__str__()


def open(filename, mode=""):

    filename = path.expand(filename)
    file = ROOT.TFile.Open(filename, mode)
    if not file:
        raise IOError("No such file: '%s'"% filename)
    file.__class__ = File
    file._path = filename
    return file
