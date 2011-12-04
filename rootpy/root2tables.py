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


def convert(rfile, hfile, rpath='', hpath='', stream=None):

    if isinstance(hfile, basestring):
        hfile = openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)

    for dirpath, dirnames, treenames in utils.walk(rfile, rpath, class_pattern='TTree'):

        if len(treenames) == 0:
            continue

        dir = utils.splitfile(dirpath)[1]
        if dir != '':
            dir = dir[1:]

        group = hfile.createGroup(hfile.root, 'root', dir)

        if stream is not None:
            print >> stream, "Will convert %i tree(s) in this directory" % len(treenames)

        for tree, treename in [(rfile.Get(os.path.join(dirpath, treename)), treename) for treename in treenames]:

            if stream is not None:
                print >> stream, "Converting %s with %i entries ..." % (treename, tree.GetEntries())
            basic_branches = []
            basic_branch_names = []
            for branch in tree.iterbranches():
                if branch.ClassName() == "TBranch":
                    basic_branches.append(branch)
                    basic_branch_names.append(branch.GetName())
            tree.SetBranchStatus('*', 0)
            fields = {}
            for branch, branch_name in zip(basic_branches, basic_branch_names):
                leaf = branch.GetListOfLeaves()[0]
                dimension = leaf.GetNdata()
                if dimension > 1:
                    if stream is not None:
                        print >> stream, "Branch %s is not a scalar. Will skip this branch." % branch
                    continue
                else:
                    type_name = leaf.GetTypeName()
                    if type_name == "Int_t":
                        fields[branch_name] = tables.Int32Col()
                    elif type_name == "UInt_t":
                        fields[branch_name] = tables.UInt32Col()
                    elif type_name == "Long64_t":
                        fields[branch_name] = tables.Int64Col()
                    elif type_name == "ULong64_t":
                        fields[branch_name] = tables.UInt64Col()
                    elif type_name == "Float_t":
                        fields[branch_name] = tables.Float32Col()
                    elif type_name == "Double_t":
                        fields[branch_name] = tables.Float64Col()
                    elif type_name == "Bool_t":
                        fields[branch_name] = tables.BoolCol()
                    else:
                        if stream is not None:
                            print >> stream, "Skipping branch %s of unsupported type: %s" % (branch_name, type_name)
                        continue
                tree.SetBranchStatus(branch_name, 1)

            if len(fields) == 0:
                if stream is not None:
                    print >> stream, "No supported branches in this tree"
                continue

            if stream is not None:
                print >> stream, "%i branch(es) will be converted" % (len(fields))

            class Event(tables.IsDescription):

                sys._getframe().f_locals.update(fields)

            table = hfile.createTable(group, treename, Event, "Event Data")
            particle = table.row
            entries = tree.GetEntries()
            prog = ProgressBar(0, entries, 37, mode='fixed')
            oldprog = str(prog)
            for i, entry in enumerate(tree):
                for name in fields.keys():
                    particle[name] = entry[name].value
                particle.append()
                prog.update_amount(i+1)
                if oldprog != str(prog):
                    print prog, "\r",
                    sys.stdout.flush()
                    oldprog = str(prog)
            print
            table.flush()
