import ROOT
from . import croot_numpy
import numpy as np


def root2array(fnames, treename, branches=None):
    """
    root2array(fnames,treename,branches=None)
    convert tree treename in root files specified in fnames to numpy structured array
    ------------------
    return numpy structure array
    fnames: list of string or string. Root file name patterns. Anything that works with TChain.Add is accepted
    treename: name of tree to convert to numpy array
    branches(optional): list of string for branch name to be extracted from tree.
    \tIf branches is not specified or is none or is empty, all from the first treebranches are extracted
    \tIf branches contains duplicate branches, only the first one is used.

    Caveat: This should not matter for most use cases. But, due to the way TChain works, if the trees specified
    in the input files have different structures, only the branch in the first tree will be automatically extracted.
    You can work around this by either reordering the input file or specifying the branches manually.
    ------------------
    Ex:
    root2array('a.root','mytree')#read all branches from tree named mytree from a.root
    root2array('a*.root','mytree')#read all branches from tree named mytree from a*.root
    root2array(['a*.root','b*.root'],'mytree')#read all branches from tree named mytree from a*.root and b*.root
    root2array('a.root','mytree','x')#read branch x from tree named mytree from a.root(useful if memory usage matters)
    root2array('a.root','mytree',['x','y'])#read branch x and y from tree named mytree from a.root
    """
    return croot_numpy.root2array(fnames, treename, branches)


def root2rec(fnames, treename, branches=None):
    """
    root2rec(fnames, treename, branches=None)
    read branches in tree treename in file(s) given by fnames can convert it to numpy recarray

    This is equivalent to root2array(fnames,treename,branches).view(np.recarray)

    see root2array for more details
    """
    return root2array(fnames, treename, branches).view(np.recarray)


def pyroot2array(tree, branches=None):
    """
    convert PyRoot TTree to numpy structured array
    see root2array for details on parameter branches
    """
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")

    if hasattr(ROOT, 'AsCapsule'):
        o = ROOT.AsCapsule(tree)
        return croot_numpy.root2array_from_capsule(o, branches)
    else:
        o = ROOT.AsCObject(tree)
        return croot_numpy.root2array_from_cobj(o, branches)


def pyroot2rec(tree, branches=None):
    """
    convert PyRoot TTree to numpy structured array
    see root2array for details on parameter branches
    """
    return pyroot2array(tree, branches).view(np.recarray)

