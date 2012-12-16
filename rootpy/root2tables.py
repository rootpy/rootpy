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
from . import log; log = log[__name__]
from .extern.progressbar import ProgressBar, Bar, ETA, Percentage
from .logger.util import check_tty

from root_numpy import tree2rec


def convert(rfile, hfile, rpath='', entries=-1):

    isatty = check_tty(sys.stdout)
    if isatty:
        widgets = [Percentage(), ' ', Bar(), ' ', ETA()]

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

        ntrees = len(treenames)
        if ntrees > 1:
            log.info("Will convert %i trees in this directory" % ntrees)
        else:
            log.info("Will convert 1 tree in this directory")

        for tree, treename in [
                (rfile.Get(os.path.join(dirpath, treename)), treename)
                for treename in treenames]:

            log.info("Converting tree '%s' with %i entries ..." % (treename,
                tree.GetEntries()))

            total_entries = tree.GetEntries()
            if isatty:
                pbar = ProgressBar(widgets=widgets, maxval=total_entries)

            if entries <= 0:
                # read the entire tree
                if isatty:
                    pbar.start()
                recarray = tree2rec(tree)
                table = hfile.createTable(
                    group, treename, recarray, tree.GetTitle())
                table.flush()
            else:
                # read the tree in chunks
                offset = 0
                while offset < total_entries:
                    if offset > 0:
                        recarray = tree2rec(tree,
                                entries=entries, offset=offset, silent=True)
                        table.append(recarray)
                    else:
                        recarray = tree2rec(tree,
                                entries=entries, offset=offset)
                        if isatty:
                            # start after any output from root_numpy
                            pbar.start()
                        table = hfile.createTable(
                            group, treename, recarray, tree.GetTitle())
                    offset += entries
                    if isatty:
                        pbar.update(offset)
                    table.flush()
            if isatty:
                pbar.finish()
