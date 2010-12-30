import string
import array
import math
import random
from rootpy.core import isbasictype
from rootpy.plotting import *
from rootpy.cut import Cut
from rootpy.style import *
import ROOT
from ROOT import gROOT, gStyle, gPad, TGraph
import os
import sys
import uuid
import operator

currentStyle = None

def readline(file, cont=None):

    line = file.readline()
    if cont != None:
        while line.strip().endswith(cont):
            line = " ".join([line.strip()[:-1*len(cont)], file.readline()])
    return line

def readlines(file, cont=None):
    
    lines = []
    line = readline(file, cont)
    while line != '':
        lines.append(line)
        line = readline(file, cont)
    return lines

def getTrees(inputFile):

    return getObjects(inputFile, "TTree")

def getTreeNames(inputFile):

    return getObjectNames(inputFile, "TTree")

def getGraphs(inputFile):
    
    return getObjects(inputFile, "TGraph")
    
def getHistos(inputFile):

    return getObjects(inputFile, "TH1D")

def getObjects(inputFile, className=""):
    
    keys = inputFile.GetListOfKeys()
    objects = []
    for key in keys:
        if className=="" or key.GetClassName() == className:
            objects.append(inputFile.Get(key.GetName()))
    return objects

def getObjectNames(inputFile, className):
    
    keys = inputFile.GetListOfKeys()
    names = []
    for key in keys:
        if key.GetClassName() == className:
            names.append(key.GetName())
    return names

def getNumEntries(trees,cuts=None,weighted=True,verbose=False):
   
    if type(trees) is not list:
        trees = [trees]
    if weighted:
        if verbose: print "Retrieving the weighted number of entries in:"
    else:
        if verbose: print "Retrieving the unweighted number of entries in:"
    wentries = 0.
    if cuts != None:
        if not cuts.empty():
            if verbose: print "Using cuts: %s"%str(cuts)
            for tree in trees:
                weight = tree.GetWeight()
                entries = tree.GetEntries(str(cuts))
                if verbose: print "%s\t%e\t%i"%(tree.GetName(),weight,entries)
                if weighted:
                    wentries += weight*entries
                else:
                    wentries += entries
            return wentries
    for tree in trees:
        weight = tree.GetWeight()
        entries = tree.GetEntries()
        if verbose: print "%s\t%e\t%i"%(tree.GetName(),weight,entries)
        if weighted:
            wentries += weight*entries
        else:
            wentries += entries
    return wentries

def makeLabel(x,y,text,textsize=-1):

    label = ROOT.TLatex(x,y,text)
    label.SetNDC()
    if textsize > 0:
        label.SetTextSize(textsize)
    return label

def drawObject(pad,object,options=""):

    pad.cd()
    object.Draw(options)
    pad.Modified()
    pad.Update()
    hold_pointers_to_implicit_members(pad)

def getTreeMaximum(trees,branchName):

    if type(trees) is not list:
        trees = [trees]
    max = -1E300
    for tree in trees:
        treeMax = tree.GetMaximum(branchName)
        if treeMax > max:
            max = treeMax
    return max 

def getTreeMinimum(trees,branchName):
    
    if type(trees) is not list:
        trees = [trees]
    min = -1E300
    for tree in trees:
        treeMin = tree.GetMinimum(branchName)
        if treeMin > min:
            min = treeMin
    return min

def drawTrees(trees,hist,expression,cuts=None,weighted=True,verbose=False):
   
    if type(trees) is not list:
        trees = [trees]
    temp_weight = 1. 
    if verbose:
        print ""
        print "Drawing the following trees onto %s:"%hist.GetName()
        print "Initial integral: %f"%hist.Integral()
    if cuts != None:
        if not cuts.empty():
            if verbose: print "cuts applied: %s"%str(cuts)
            for tree in trees:
                if verbose: print tree.GetName()
                if not weighted:
                    temp_weight = tree.GetWeight()
                    tree.SetWeight(1.)
                tree.Draw("%s>>+%s"%(expression,hist.GetName()),str(cuts))
                if not weighted:
                    tree.SetWeight(temp_weight)
            if verbose:
                print "Final integral: %f"%hist.Integral()
            return
    for tree in trees:
        if verbose: print tree.GetName()
        if not weighted:
            temp_weight = tree.GetWeight()
            tree.SetWeight(1.)
        tree.Draw("%s>>+%s"%(expression,hist.GetName()))
        if not weighted:
            tree.SetWeight(temp_weight)
    if verbose:
        print "Final integral: %f"%hist.Integral()

def closest(target, collection):

    return collection.index((min((abs(target - i), i) for i in collection)[1]))

def round_to_n(x, n):

    if n < 1:
        raise ValueError("number of significant digits must be >= 1")
    return "%.*g" % (n, x)

def drawLogGraphs(pad,graphs,title,xtitle,ytitle,legend=None,legendheight=1.,label=None,format="png"):
    
    pad.cd()
    pad.SetLogy()
    #if format not in ("pdf","eps"):
    #pad.SetGrid()
    
    if not legend:
        legend,legendheight = getLegend(len(graphs),pad)
    #legend.SetEntrySeparation(0.01)
    
    xmax = -1E20
    xmin = 1E20
    ymax = -1E20
    ymin = 1E20
    for graph in graphs:
        txmax = graph.xMax()
        txmin = graph.xMin()
        tymax = graph.yMax()
        tymin = graph.yMin()
        if txmax > xmax:
            xmax = txmax
        if txmin < xmin:
            xmin = txmin
        if tymax > ymax:
            ymax = tymax
        if tymin < ymin:
            ymin = tymin
        
    for index,graph in enumerate(graphs):
        legend.AddEntry(graph,graph.GetTitle(),"P")
        graph.SetMarkerSize(1.5)
        if index==0:
            graph.SetTitle(title)
            graph.GetXaxis().SetLimits(xmin,xmax)
            graph.GetXaxis().SetRangeUser(xmin,xmax)
            graph.GetXaxis().SetTitle(xtitle)
            graph.GetYaxis().SetLimits(ymin,ymax)
            graph.GetYaxis().SetRangeUser(ymin,ymax)
            graph.GetYaxis().SetTitle(ytitle)
            graph.Draw("AP")
        else:
            graph.Draw("P SAME")
    
    legend.Draw()
    if label:
        label.Draw()
    pad.Modified()
    pad.Update()
    for item in pad.GetListOfPrimitives():
        if isinstance(item,ROOT.TPaveText):
            text = item.GetLine(0)
            text.SetTextFont(63)
            text.SetTextSizePixels(20)
    _hold_pointers_to_implicit_members(pad)

def draw(
        hists,
        pad = None,
        title = None,
        axislabels = None,
        legend = None,
        showlegend = True,
        textlabels = None,
        xscale = "linear",
        yscale = "linear",
        minimum = None,
        maximum = None,
        use_global_margins=True
    ):
    
    if type(hists) is not list:
        hists = [hists]

    hists = [hist.Clone() for hist in hists]
    
    if axislabels:
        if type(axislabels) is not list:
            axislabels = [axislabels]
    
    if textlabels:
        if type(textlabels) is not list:
            textlabels = [textlabels]

    if not pad:
        pad = ROOT.TCanvas(uuid.uuid4().hex,uuid.uuid4().hex,0,0,800,600)
    
    pad.cd()

    if yscale == "log":
        pad.SetLogy(True)
    else:
        pad.SetLogy(False)
    if xscale == "log":
        pad.SetLogx(True)
    else:
        pad.SetLogx(False)
    
    if use_global_margins:
        pad.SetTopMargin(ROOT.gStyle.GetPadTopMargin())
        pad.SetRightMargin(ROOT.gStyle.GetPadRightMargin())
        pad.SetBottomMargin(ROOT.gStyle.GetPadBottomMargin())
        pad.SetLeftMargin(ROOT.gStyle.GetPadLeftMargin())

    if title:
        pad.SetTopMargin(0.1)

    for hist in hists:
        if isinstance(hist, plotting.HistStack):
            subhists = hist
        else:
            subhists = [hist]
        for subhist in subhists:
            if "colz" in subhist.format.lower():
                if not title:
                    pad.SetTopMargin(0.06)
                pad.SetRightMargin(0.13)
                break

    nhists = 0
    for hist in hists:
        if isinstance(hist, plotting.HistStack):
            nhists += len(hist)
        else:
            nhists += 1
    
    if not legend and showlegend:
        legend = Legend(nhists, pad)
    elif not showlegend:
        legend = None
    
    for hist in histos:
        if hist.norm:
            if isinstance(hist.norm, plotting._HistBase) or isinstance(hist.norm, plotting.HistStack):
                norm = hist.norm.Integral()
                integral = hist.Integral()
            elif hist.norm.lower() == "max":
                norm = 1.
                integral = hist.GetMaximum()
            elif isbasictype(hist.norm):
                norm = hist.norm
                integral = hist.Integral()
            if integral > 0:
                hist.Scale(norm / integral)
    
    _max = None  # negative infinity
    _min = ()    # positive infinity
    for hist in histos:
        lmax = hist.GetMaximum(includeError=True)
        lmin = hist.GetMinimum(includeError=True)
        if lmax > _max:
            _max = lmax
        if lmin < _min and not (yscale == "log" and lmin <= 0.):
            _min = lmin

    if maximum != None:
        if maximum > _max:
            _max = maximum
    if minimum != None:
        if minimum < _min:
            _min = minimum
    
    if legend:
        plotheight = 1 - pad.GetTopMargin() - pad.GetBottomMargin()
        legendheight = legend.Height() + padding
        padding = 0.05
        if yscale == "linear":
            _max = (_max - (_min * legendheight / plotheight)) / (1. - (legendheight / plotheight))
        else: # log
            if _max <= 0.:
                raise ValueError("Attempted to plot log scale where max<=0: %f"% _max)
            if _min <= 0.:
                raise ValueError("Attempted to plot log scale where min<=0: %f"% _min)
            _max = 10.**((math.log10(_max) - (math.log10(_min) * legendheight / plotheight)) / (1. - (legendheight / plotheight)))
    else:
        _max += (_max - _min)*.1

    if _min > 0 and _min - (_max - _min)*.1 < 0:
        _min = 0. 
    
    for index,hist in enumerate(hists):
       
        if legend:
            legend.AddEntry(hist) 
        drawOptions = []
        if index == 0 or not axesDrawn:
            hist.Draw()
            if hist.visible:
                axesDrawn = True
            hist.SetTitle(title)
            hist.GetXaxis().SetTitle(axislabels[0])
            hist.GetYaxis().SetTitle(axislabels[1])
            if _max > _min and len(axislabels) == 2:
                hist.GetYaxis().SetLimits(_min, _max)
                hist.GetYaxis().SetRangeUser(_min, _max)
            if _max > _min and len(axislabels) == 3:
                hist.GetZaxis().SetLimits(_min, _max)
                hist.GetZaxis().SetRangeUser(_min, _max)
            if hist.intMode:
                hist.GetXaxis().SetNdivisions(len(hist),True)
            if len(axislabels) >= 3:
                hist.GetZaxis().SetTitle(axislabels[2])
                if len(axislabels) == 4:
                    hist.SetTitle(axislabels[3])
                    hist.GetZaxis().SetTitleOffset(1.8)
        else:
            hist.SetTitle("")
            hist.Draw("same")
    
    if legend:
        legend.Draw()

    for label in textlabels:
        label.Draw()

    pad.Modified()
    pad.Update()
    for item in pad.GetListOfPrimitives():
        if isinstance(item, ROOT.TPaveText):
            text = item.GetLine(0)
            text.SetTextFont(63)
            text.SetTextSizePixels(20)
    _hold_pointers_to_implicit_members(pad)
    return pad

def save_pad(pad,filename=None,format="png",dir=None):
    
    if not filename:
        filename = pad.GetName() #To Fix
    for c in string.punctuation:
        filename = filename.replace(c,'-')
    filename = filename.strip().replace(' ','-')
    
    if dir:
        filename = dir.strip("/")+"/"+filename
    
    formats = format.split('+')
    for imageformat in formats:
        pad.Print(".".join([filename,imageformat]))

def _hold_pointers_to_implicit_members( obj ):
    
    if not hasattr(obj, '_implicit_members'):
        obj._implicit_members = []
    if hasattr(obj, 'GetListOfPrimitives'):
        for prim in obj.GetListOfPrimitives():
            if prim not in obj._implicit_members:
                obj._implicit_members.append(prim)

def ROOTlogon(batch=False,noGlobal=False,style="MINE"):

    global currentStyle
    if noGlobal:
        ROOT.TH1.AddDirectory(False) # Stupid global variables in ROOT... doing this will screw up TTree.Draw()
    if batch:
        ROOT.gROOT.SetBatch()
    ROOT.TH1.SetDefaultSumw2(True)
    #ROOT.gROOT.SetStyle("Plain")
    ROOT.TGaxis.SetMaxDigits(3)
    tstyle = getStyle(style)
    currentStyle = tstyle
    if tstyle:
        print "Using ROOT style %s"%tstyle.GetName()
        ROOT.gROOT.SetStyle(tstyle.GetName())
        ROOT.gROOT.ForceStyle()
        ROOT.gStyle.SetPalette(1)
    else:
        print "Style %s is not defined"%style
