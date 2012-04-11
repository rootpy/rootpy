import _librootnumpy
import numpy as np
import sys
from glob import glob
try:
    import ROOT
except ImportError:
    pass


def list_trees(fname):
    return _librootnumpy.list_trees(fname)


def list_branches(fname, treename):
    return _librootnumpy.list_branches(fname, treename)


def root2array(fnames, treename=None, branches=None):
    """
    root2array(fnames, treename, branches=None)
    convert tree treename in root files specified in fnames to
    numpy structured array
    ------------------
    return numpy structure array
    fnames: list of string or string. Root file name patterns.
    Anything that works with TChain.Add is accepted
    treename: name of tree to convert to numpy array.
    This is optional if the file contains exactly 1 tree.
    branches(optional): list of string for branch name to be
    extracted from tree.
    * If branches is not specified or is none or is empty,
      all from the first treebranches are extracted
    * If branches contains duplicate branches, only the first one is used.

    Caveat: This should not matter for most use cases. But, due to
    the way TChain works, if the trees specified
    in the input files have different structures, only the
    branch in the first tree will be automatically extracted.
    You can work around this by either reordering the input
    file or specifying the branches manually.
    ------------------
    Ex:
    # read all branches from tree named mytree from a.root
    root2array('a.root', 'mytree')
    # read all branches from tree named mytree from a*.root
    root2array('a*.root', 'mytree')
    # read all branches from tree named mytree from a*.root and b*.root
    # read branch x from tree named mytree from
    # a.root(useful if memory usage matters)
    root2array(['a*.root', 'b*.root'], 'mytree')
    # read branch x from tree named mytree from
    # a.root(useful if memory usage matters)
    root2array('a.root', 'mytree', 'x')
    #read branch x and y from tree named mytree from a.root
    root2array('a.root', 'mytree', ['x', 'y'])
    """
    if treename is None:
        afname = None
        if isinstance(fnames, basestring):
            afname = glob(fnames)
        else:
            afname = glob(fnames[0])
        trees = list_trees(afname[0])
        if len(trees) != 1:
            raise ValueError('treename need to be specified if the file '
                             'contains more than 1 tree. Your choices are:'
                             + str(trees))
        else:
            treename = trees[0]
    return _librootnumpy.root2array(fnames, treename, branches)


def root2rec(fnames, treename=None, branches=None):
    """
    root2rec(fnames, treename, branches=None)
    read branches in tree treename in file(s) given by fnames can
    convert it to numpy recarray

    This is equivalent to
    root2array(fnames, treename, branches).view(np.recarray)

    see root2array for more details
    """
    if treename is None:
        afname = None
        if isinstance(fnames, basestring):
            afname = glob(fnames)
        else:
            afname = glob(fnames[0])
        trees = list_trees(afname[0])
        if len(trees) != 1:
            raise ValueError('treename need to be specified if the file '
                             'contains more than 1 tree. Your choices are:'
                             + str(trees))
        else:
            treename = trees[0]
    return root2array(fnames, treename, branches).view(np.recarray)


def tree2array(tree, branches=None):
    """
    convert PyRoot TTree to numpy structured array
    see root2array for details on parameter branches
    """
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")

    if hasattr(ROOT, 'AsCapsule'):
        o = ROOT.AsCapsule(tree)
        return _librootnumpy.root2array_from_capsule(o, branches)
    else:
        o = ROOT.AsCObject(tree)
        return _librootnumpy.root2array_from_cobj(o, branches)


def tree2rec(tree, branches=None):
    """
    convert PyRoot TTree to numpy structured array
    see root2array for details on parameter branches
    """
    return tree2array(tree, branches).view(np.recarray)
