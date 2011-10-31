import string
import array
import math
import random
from .core import isbasictype
from .plotting import *
from .plotting.core import dim
from .plotting.hist import _HistBase
from .tree import Cut
from .plotting.style import *
import ROOT
from ROOT import gROOT, gStyle, gPad, TGraph
import os
import sys
import uuid
import operator

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

def getNumEntriesWeightedSelection(trees,cuts,weighted=True,branch=None,minimum=None,maximum=None,verbose=False):
   
    if type(trees) not in [list, tuple]:
        trees = [trees]
    if weighted:
        if verbose: print "Retrieving the weighted number of entries (with weighted selection) in:"
    else:
        if verbose: print "Retrieving the unweighted number of entries (with weighted selection) in:"
    wentries = 0.
    if verbose: print "Using cuts: %s"%str(cuts)
    for tree in trees:
        if branch is None:
            branch = tree.GetListOfBranches()[0].GetName()
        if "EventNumber" in branch:
            branch = tree.GetListOfBranches()[1].GetName()
        if minimum is None:
            minimum = getTreeMinimum(tree, branch)
        if maximum is None:
            maximum = getTreeMaximum(tree, branch)
        if minimum == maximum:
            minimum -= 1
            maximum += 1
        if verbose: print "using branch %s with min %f and max %f"% (branch, minimum, maximum)
        width = maximum - minimum
        minimum -= width/2
        maximum += width/2
        hist = Hist(1,minimum,maximum)
        draw_trees(trees = tree, expression = branch, hist = hist, cuts = cuts, weighted = weighted)
        entries = hist.Integral()
        wentries += entries
        if verbose: print "%s\t%e\t%f"%(tree.GetName(),tree.GetWeight(),entries)
    return wentries

def getNumEntries(trees,cuts=None,weighted=True,verbose=False):
   
    if type(trees) not in [list, tuple]:
        trees = [trees]
    if weighted:
        if verbose: print "Retrieving the weighted number of entries in:"
    else:
        if verbose: print "Retrieving the unweighted number of entries in:"
    wentries = 0.
    if cuts != None:
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

def makeLabel(x, y, text, size = None, font = None):

    label = ROOT.TLatex(x,y,text)
    label.SetNDC()
    if size is not None:
        label.SetTextSize(size)
    if font is not None:
        label.SetTextFont(font)
    return label

def drawObject(pad,object,options=""):

    pad.cd()
    object.Draw(options)
    pad.Modified()
    pad.Update()
    hold_pointers_to_implicit_members(pad)

def getTreeMaximum(trees, expression, cut = None):

    if type(trees) not in [list, tuple]:
        trees = [trees]
    _max = None # - infinity
    for tree in trees:
        treeMax = tree.GetMaximum(expression, cut)
        if treeMax > _max:
            _max = treeMax
    return _max 

def getTreeMinimum(trees, expression, cut = None):
    
    if type(trees) not in [list, tuple]:
        trees = [trees]
    _min = () # + infinity
    for tree in trees:
        treeMin = tree.GetMinimum(expression, cut)
        if treeMin < _min:
            _min = treeMin
    return _min

def draw_samples(
        samples,
        expression,
        hist = None,
        cuts = None,
        weighted = True,
        verbose = False
    ):

    if type(samples) is not list:
        samples = [samples]
    trees = reduce(operator.add, [sample.trees for sample in samples])
    return draw_trees(
        trees,
        expression,
        hist,
        cuts,
        weighted,
        verbose)

def draw_trees(
        trees,
        expression,
        hist = None,
        cuts = None,
        weighted = True,
        verbose = False
    ):
   
    if type(trees) is not list:
        trees = [trees]
    if hist is not None:
        histname = hist.GetName()
    else:
        histname = uuid.uuid4().hex
    temp_weight = 1. 
    if verbose:
        print ""
        print "Drawing the following trees onto %s:"% histname
        if hist is not None:
            print "Initial integral: %f"% hist.Integral()
    if cuts:
        if verbose: print "cuts applied: %s"%str(cuts)
    for tree in trees:
        if verbose: print tree.GetName()
        if not weighted:
            temp_weight = tree.GetWeight()
            tree.SetWeight(1.)
        if cuts:
            ohist = tree.Draw("%s>>+%s"%(expression,histname),str(cuts))
        else:
            ohist = tree.Draw("%s>>+%s"%(expression,histname))
        if not weighted:
            tree.SetWeight(temp_weight)
    if verbose:
        print "Final integral: %f"%hist.Integral()
    return ohist

def closest(target, collection):

    return collection.index((min((abs(target - i), i) for i in collection)[1]))

def round_to_n(x, n):

    if n < 1:
        raise ValueError("number of significant digits must be >= 1")
    return "%.*g" % (n, x)


def ratioPlot(graphs, reference):

    ratios = [Graph.divide(graph, reference, consistency=False) for graph in graphs]
    return ratios

def drawGraphs(pad,
               graphs,
               title,
               xtitle,
               ytitle,
               legend=None,
               label=None,
               format="png",
               xmin=None,
               xmax=None,
               ymin=None,
               ymax=None,
               yscale="log"):
    
    if xmin is None:
        xmin = ()
    if ymin is None:
        ymin = ()
    
    pad.cd()
    if yscale == "log":
        pad.SetLogy()
    
    if not legend:
        legend = Legend(len(graphs),pad)
    
    lxmin, lymin = (), ()
    lxmax, lymax = None, None
    for graph in graphs:
        txmax = graph.xMax()
        txmin = graph.xMin()
        tymax = graph.yMax()
        tymin = graph.yMin()
        if txmax > lxmax:
            lxmax = txmax
        if txmin < lxmin:
            lxmin = txmin
        if tymax > lymax:
            lymax = tymax
        if tymin < lymin:
            lymin = tymin

    if xmin is ():
        xmin = lxmin
    if xmax is None:
        xmax = lxmax
    if ymin is ():
        ymin = lymin
    if ymax is None:
        ymax = lymax
        
    for index,graph in enumerate(graphs):
        legend.AddEntry(graph)
        graph.SetMarkerSize(1.5)
        if index==0:
            graph.SetTitle(title)
            graph.GetXaxis().SetLimits(xmin,xmax)
            graph.GetXaxis().SetRangeUser(xmin,xmax)
            graph.GetXaxis().SetTitle(xtitle)
            graph.GetYaxis().SetLimits(ymin,ymax)
            graph.GetYaxis().SetRangeUser(ymin,ymax)
            graph.GetYaxis().SetTitle(ytitle)
            graph.Draw('A')
        else:
            graph.Draw('SAME')
    
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
        objects,
        pad = None,
        title = None,
        axislabels = None,
        legend = None,
        showlegend = True,
        greedylegend = False,
        textlabels = None,
        xscale = "linear",
        yscale = "linear",
        style2d = "col",
        style3d = "surf1",
        maxmin = (),
        minmax = None,
        minimum = 0,
        maximum = None,
        use_global_margins = True
    ):
    
    if type(objects) not in [list, tuple]:
        objects = [objects]

    objects = [hist.Clone() for hist in objects]
   
    dimension = None
    for thing in objects:
        if dimension is None:
            dimension = dim(thing)
        elif dim(thing) != dimension:
            raise TypeError("dimensions of histograms must all be the same")
    
    if axislabels is not None:
        if type(axislabels) not in [list, tuple]:
            axislabels = [axislabels]
    else:
        axislabels = []
    
    if textlabels is not None:
        if type(textlabels) not in [list, tuple]:
            textlabels = [textlabels]
    else:
        textlabels = []

    if pad is None:
        pad = Canvas(uuid.uuid4().hex,"Canvas",0,0,800,600)
    else:
        pad.Clear()
    
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
    else:
        title = ""

    for hist in objects:
        if isinstance(hist, HistStack):
            subobjects = hist
        else:
            subobjects = [hist]
        for subhist in subobjects:
            if "colz" in subhist.format.lower():
                if not title:
                    pad.SetTopMargin(0.06)
                pad.SetRightMargin(0.13)
                break

    nobjects = 0
    for hist in objects:
        if isinstance(hist, HistStack):
            nobjects += len(hist)
        else:
            nobjects += 1
    
    if not legend and showlegend:
        legend = Legend(nobjects, pad)
        legend.SetTextSize(20)
    
    for hist in objects:
        if hist.norm:
            if isinstance(hist.norm, _HistBase) or isinstance(hist.norm, HistStack):
                norm = hist.norm.Integral()
                integral = hist.Integral()
            elif type(hist.norm) is str:
                if hist.norm.lower() == "max":
                    norm = 1.
                    integral = hist.GetMaximum()
                else:
                    raise ValueError("Normalization not understood: %s"% hist.norm)
            elif isbasictype(hist.norm):
                norm = hist.norm
                integral = hist.Integral()
            if integral > 0:
                hist.Scale(norm / integral)
    
    _max = None  # negative infinity
    _min = ()    # positive infinity
    for hist in objects:
        if dim(hist) == 1:
            lmax = hist.GetMaximum(include_error=True)
            lmin = hist.GetMinimum(include_error=True)
        else:
            lmax = hist.GetMaximum()
            lmin = hist.GetMinimum()
        if lmax > _max:
            _max = lmax
        if lmin < _min and not (yscale == "log" and lmin <= 0.):
            _min = lmin

    __max, __min = _max, _min

    if maximum != None:
        if maximum > _max:
            _max = maximum
    if minimum != None:
        if minimum < _min and not (yscale == "log" and minimum <= 0.):
            _min = minimum

    if _min > maxmin:
        _min = maxmin
    if _max < minmax:
        _max = minmax
    
    if legend and greedylegend:
        padding = 0.05
        plotheight = (1 - pad.GetTopMargin()) - pad.GetBottomMargin()
        legendheight = legend.Height() + padding
        if yscale == "linear":
            _max = (_max - (_min * legendheight / plotheight)) / (1. - (legendheight / plotheight))
        else: # log
            if _max <= 0.:
                raise ValueError("Attempted to plot log scale where max<=0: %f"% _max)
            if _min <= 0.:
                raise ValueError("Attempted to plot log scale where min<=0: %f"% _min)
            _max = 10.**((math.log10(_max) - (math.log10(_min) * legendheight / plotheight)) / (1. - (legendheight / plotheight)))
    else:
        if yscale == "linear":
            if maximum is None:
                _max += (_max - _min)*.1
            if _min != 0:
                _min -= (_max - _min)*.1
        else:
            height = math.log10(_max) - math.log10(_min)
            if maximum is None:
                _max *= 10**(height*.1)
            if _min != 0:
                _min *= 10**(height*-.1)

    format = ""
    if len(axislabels)==3:
        format += style2d
    elif len(axislabels)==4:
        format += style3d

    for index,hist in enumerate(objects):       
        if legend:
            legend.AddEntry(hist)
        if index == 0 or not axesDrawn:
            if title:
                hist.SetTitle(title)
            else:
                hist.SetTitle("")
            if isinstance(hist, Graph):
                hist.Draw('AP',format)
            else:
                hist.Draw(format)
            if hist.visible:
                axesDrawn = True
            if axislabels:
                hist.GetXaxis().SetTitle(axislabels[0])
                if len(axislabels) > 1:
                    hist.GetYaxis().SetTitle(axislabels[1])
                if len(axislabels) >= 3:
                    hist.GetZaxis().SetTitle(axislabels[2])
                    if len(axislabels) == 4:
                        hist.SetTitle(axislabels[3])
                        hist.GetZaxis().SetTitleOffset(1.8)
            if _max > _min and dimension in (1, 2):
                hist.GetYaxis().SetLimits(_min, _max)
                hist.GetYaxis().SetRangeUser(_min, _max)
            if _max > _min and dimension == 3:
                hist.GetZaxis().SetLimits(_min, _max)
                hist.GetZaxis().SetRangeUser(_min, _max)
            if hist.intmode:
                hist.GetXaxis().SetNdivisions(len(hist),True)
        else:
            hist.SetTitle("")
            if isinstance(hist, Graph):
                hist.Draw("P same",format)
            else:
                hist.Draw("same",format)
    
    if legend:
        legend.Draw()

    for label in textlabels:
        label.Draw()

    """
    for item in pad.GetListOfPrimitives():
        if isinstance(item, ROOT.TPaveText):
            text = item.GetLine(0)
            text.SetTextFont(63)
            text.SetTextSizePixels(20)
    """
    pad.OwnMembers()
    pad.Modified()
    pad.Update()
    return pad, __max, __min

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

def animate_pads(pads, filename = None, loop = True, delay = 50):
    
    if type(pads) not in [list, tuple]:
        pads = [pads]
    if filename is None:
        filename = pads[0].GetName()
    for frameindex,pad in enumerate(pads):
        framename = "%s_%i.png"% (pad.GetName(), frameindex)
        frames.append(framename)
        pad.Print(framename)
    frame_args = " ".join(frames)
    if os.system("convert -delay %i -loop %i %s %s"%(delay, loop, frame_args, filename+".gif")) != 0:
        raise RuntimeError("Could not create animation. Is ImageMagick installed?")
    for frame in frames:
        os.unlink(frame)

def _hold_pointers_to_implicit_members( obj ):
    
    if not hasattr(obj, '_implicit_members'):
        obj._implicit_members = []
    if hasattr(obj, 'GetListOfPrimitives'):
        for prim in obj.GetListOfPrimitives():
            if prim not in obj._implicit_members:
                obj._implicit_members.append(prim)

def set_style(style):

    print "Using ROOT style %s" % style.GetName()
    ROOT.gROOT.SetStyle(style.GetName())
    ROOT.gROOT.ForceStyle()
    ROOT.gStyle.SetPalette(1)

def logon(batch=True, style=None):

    if batch:
        ROOT.gROOT.SetBatch()
    ROOT.TH1.SetDefaultSumw2(True)
    #ROOT.gROOT.SetStyle("Plain")
    ROOT.TGaxis.SetMaxDigits(3)
    if style is not None:
        set_style(style)
