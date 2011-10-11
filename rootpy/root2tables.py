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
from . import types as rtypes
from .utils.progressbar import *
from .io import open as ropen, utils

def convert(rfile, hfile, rpath='', hpath='', stream=sys.stdout):
    
    if isinstance(hfile, basestring):
        hfile = openFile(filename=hfile, mode="w", title="Data")
    if isinstance(rfile, basestring):
        rfile = ropen(rfile)
     
    for dirpath, dirnames, treenames in utils.walk(rfile, rpath, pattern='TTree'):

        if len(treenames) = 0:
            continue

        if path == "":
            group = "/"
        else:
            group = path
        
        if directory != "":
            print >> stream, "Creating group %s" % directory
            group = hd5File.createGroup(group, directory, directory)
        print >> stream, "Will convert %i trees in this directory" % len(treenames)
        
        for tree,treeName in [(currDir.Get(treeName),treeName) for treeName in trees]:

            print >> stream, "Converting %s with %i entries ..."%(tree.GetName(),tree.GetEntries())
            branches = tree.GetListOfBranches()
            basicBranches = []
            for branch in branches:
                if branch.ClassName() == "TBranch":
                    basicBranches.append(branch)
            tree.SetBranchStatus("*",0)
            fields = {}
            valueMap = {}
            for branch in basicBranches:
                skip = False
                fieldName = branch.GetName()
                leaf = branch.GetListOfLeaves()[0]
                dimension = leaf.GetNdata()
                if leaf.GetNdata() > 1:
                    print >> stream, "Branch %s is not a scalar. Will skip this branch."% branch
                    skip = True
                else:
                    typeName = leaf.GetTypeName()
                    if typeName == "Int_t":
                        fields[fieldName]=Int32Col()
                        valueMap[fieldName]=Int()
                    elif typeName == "UInt_t":
                        fields[fieldName]=UInt32Col()
                        valueMap[fieldName]=UInt()
                    elif typeName == "Float_t":
                        fields[fieldName]=Float32Col()
                        valueMap[fieldName]=Float()
                    elif typeName == "Double_t":
                        fields[fieldName]=Float64Col()
                        valueMap[fieldName]=Double()
                    elif typeName == "Bool_t":
                        fields[fieldName]=BoolCol()
                        valueMap[fieldName]=Bool()
                    else:
                        print >> stream, "Skipping branch %s of unsupported type: %s"%(fieldName,typeName)
                        skip = True                        
                if not skip:
                    tree.SetBranchStatus(fieldName,1)
                    tree.SetBranchAddress(fieldName,valueMap[fieldName])
            if len(fields) == 0:
                print >> stream, "No supported branches in this tree"
                continue

            print >> stream, "%i total branches" % (len(fields))

            class Event(IsDescription):
                
                sys._getframe().f_locals.update(fields)
           
            table = hd5File.createTable(group,treeName,Event,"Event Data")
            particle = table.row
            entries = tree.GetEntries()
            prog = ProgressBar(0,entries,37,mode='fixed')
            oldprog = str(prog)
            for i in xrange(entries):
                tree.GetEntry(i)
                for name in fields.keys():
                    particle[name] = valueMap[name].value()
                particle.append()
                prog.update_amount(i+1)
                if oldprog != str(prog):
                    print prog, "\r",
                    sys.stdout.flush()
                    oldprog = str(prog)
            print
            table.flush()
