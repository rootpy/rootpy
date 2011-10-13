"""
This module should handle:
* conversion of ROOT's TFile and contained TTrees into HDF5 format with PyTables
  A first attempt is in scripts/root2hd5
"""

import sys
import os
import traceback
import numpy
import tables
import ROOT
from .utils.progressbar import *
from .io import open as ropen, utils

def convert(rfile, hfile, rpath='', hpath='', stream=sys.stdout):
    
    if isinstance(hfile, basestring):
        hfile = openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)
     
    for dirpath, dirnames, treenames in utils.walk(rfile, rpath, pattern='TTree'):

        if len(treenames) == 0:
            continue

        group = utils.splitfile(dirpath)[2]
        if group == ''
            group = '/'
         
        if directory != "":
            print >> stream, "Creating group %s" % directory
            group = hd5File.createGroup(group, directory, directory)
        print >> stream, "Will convert %i trees in this directory" % len(treenames)
        
        for tree, treename in [(rfile.Get(os.path.join(dirpath, treename), treename) for treename in treenames]:

            print >> stream, "Converting %s with %i entries ..."%(treename, tree.GetEntries())
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
                    print >> stream, "Branch %s is not a scalar. Will skip this branch." % branch
                    continue
                else:
                    type_name = leaf.GetTypeName()
                    if type_name == "Int_t":
                        fields[fieldName]=Int32Col()
                    elif type_name == "UInt_t":
                        fields[fieldName]=UInt32Col()
                    elif type_name == "Long64_t":
                        fields[fieldName]=Int64Col()
                    elif type_name == "ULong64_t":
                        fields[fieldName]=UInt64Col()
                    elif type_name == "Float_t":
                        fields[fieldName]=Float32Col()
                    elif type_name == "Double_t":
                        fields[fieldName]=Float64Col()
                    elif type_name == "Bool_t":
                        fields[fieldName]=BoolCol()
                    else:
                        print >> stream, "Skipping branch %s of unsupported type: %s" % (branch_name, type_name)
                        continue
                 tree.SetBranchStatus(branch_name, 1)

            if len(fields) == 0:
                print >> stream, "No supported branches in this tree"
                continue

            print >> stream, "%i branches will be converted" % (len(fields))

            class Event(IsDescription):
                
                sys._getframe().f_locals.update(fields)
           
            table = hd5File.createTable(group, treename, Event, "Event Data")
            particle = table.row
            entries = tree.GetEntries()
            prog = ProgressBar(0, entries, 37, mode='fixed')
            oldprog = str(prog)
            for entry in tree:
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
