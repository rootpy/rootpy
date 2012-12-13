# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
This module handles conversion of ROOT's TFile and
contained TTrees into HDF5 format with PyTables

Also see scripts/root2hd5
"""
import os
import sys
import tables
from .io import open as ropen, utils
from root_numpy import tree2rec


def convert(rfile, hfile, rpath='', stream=sys.stdout):

    if isinstance(hfile, basestring):
        hfile = tables.openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)

    for dirpath, dirnames, treenames in utils.walk(
            rfile, rpath, class_pattern='TTree'):

        # skip root
        if not dirpath and not treenames:
            continue

        # skip directories w/o trees or subdirs
        if not dirnames and not treenames:
            continue

        where_group = '/' + os.path.dirname(dirpath)
        current_dir = os.path.basename(dirpath)

        if not current_dir:
            group = hfile.root
        else:
            group = hfile.createGroup(where_group, current_dir, "")

        if stream is not None:
            print >> stream, "Will convert %i tree(s) in this directory" % \
                    len(treenames)

        for tree, treename in [
                (rfile.Get(os.path.join(dirpath, treename)), treename)
                for treename in treenames]:

            if stream is not None:
                print >> stream, "Converting %s with %i entries ..." % \
                        (treename, tree.GetEntries())

            recarray = tree2rec(tree)
            table = hfile.createTable(
                    group, treename, recarray, tree.GetTitle())
            table.flush()
