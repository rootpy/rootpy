"""
This module handles conversion of ROOT's TFile and
contained TTrees into HDF5 format with PyTables

Also see scripts/root2hd5
"""

import sys
import os
import traceback
import numpy
import tables
import ROOT
from .progressbar import *
from .io import open as ropen, utils
from .tree import Tree
from .root2array import tree_to_recarray

def convert(rfile, hfile, rpath='', stream=sys.stdout):

    if isinstance(hfile, basestring):
        hfile = tables.openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)

    for dirpath, dirnames, treenames in utils.walk(rfile, rpath, class_pattern='TTree'):

        if not dirpath and not treenames: # skip root
            continue

        if not dirnames and not treenames: # skip directories w/o trees or subdirs
            continue

        where_group = '/' + os.path.dirname(dirpath)
        current_dir = os.path.basename(dirpath)

        if where_group == '/':
            group = hfile.root
        else:
            group = hfile.createGroup(where_group, current_dir, "")

        if stream is not None:
            print >> stream, "Will convert %i tree(s) in this directory" % len(treenames)

        for tree, treename in [(rfile.Get(os.path.join(dirpath, treename)), treename) for treename in treenames]:

            if stream is not None:
                print >> stream, "Converting %s with %i entries ..." % (treename, tree.GetEntries())

            recarray = tree_to_recarray(tree, None, False)
            table = hfile.createTable(group, treename, recarray, tree.GetTitle())
            table.flush()
