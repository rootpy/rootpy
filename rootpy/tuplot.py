import sys
import os
import ROOT
import re
import uuid
import traceback
from rootpy.datamanaging import *
from rootpy import metadata
from rootpy.cut import Cut
from rootpy import parsing
from rootpy.tree import *
from rootpy.plotting import *
from rootpy import routines
from rootpy.style import markers, colors, fills
from rootpy import measure
from array import array

manager = DataManager(verbose=False)
routines.ROOTlogon(style="ATLAS", batch = sys.stdout.isatty())
blankCanvas = True

properties = {"title"       : {"type":"str","value":""},
              "label"       : {"type":"str","value":""},
              "labelx"      : {"type":"float","value":0.2},
              "labely"      : {"type":"float","value":0.8},
              "textsize"    : {"type":"float","value":20},
              "ylabel"      : {"type":"str","value":"Entries"},
              "normalize"   : {"type":"str","value":"NONE"},
              "linewidth"   : {"type":"int","value":2},
              "markersize"  : {"type":"float","value":2.},
              "yscale"      : {"type":"str","value":"linear"},
              "xscale"      : {"type":"str","value":"linear"},
              "canvaswidth" : {"type":"int","value":800},
              "canvasheight": {"type":"int","value":600},
              "bins"        : {"type":"int","value":50},
              "showunits"   : {"type":"bool","value":True},
              "showbinsize" : {"type":"bool","value":True},
              "showlegend"  : {"type":"bool","value":True},
              "legendmode"  : {"type":"bool","value":True},
              "imageformat" : {"type":"str","value":"png"}}

canvas = Canvas(uuid.uuid4().hex,uuid.uuid4().hex,0,0,properties["canvaswidth"]["value"],properties["canvasheight"]["value"])

objects = {"legend": True,
           "cut":    False,
           "label":  True}

dictionary = {}
localVariableMeta = {}
plotMode = "default"

def plot(sampledicts,expression,cuts,reference=None,norm=None,stacked=None):
    
    global manager
    global canvas
    global properties
    global objects
    global blankCanvas
    global dictionary
    normHist = None
    try:
        cuts = Cut(cuts)
    except:
        error("cut expression is not well-formed")
        return None
    
    samples = []
    for sample in sampledicts:
        samplelist = manager.get_samples(sample["sample"], properties = sample)
        for subsample in samplelist:
            if not subsample:
                print "sample %s not found"%(sample["name"])
                return None
            if not subsample.trees:
                print "sample %s not found"%(sample["name"])
                return None
        samples.append(samplelist)
   
    # take meta from first sample
    meta = samples[0][0].meta
     
    intmode = False
    varExpr = expression.split(',')
    varExpr.reverse() # Why ROOT... WHY???
    variables = []
    for var in varExpr:
        binlist = None
        if ":" in var:
            var,binlist = var.split(":")
        if localVariableMeta.has_key(var):
            details = localVariableMeta[var]
        else:
            details = metadata.get_variable_meta(var,meta)
            addVariable(var,details)
            if not details:
                details = localVariableMeta[var]
        info = {}
        info.update(details)
        xmin,xmax = details['range']
        
        if details["type"].lower()=='int':
            intmode = True
        elif samplelist[0].properties.has_key("intmode"):
            if samplelist[0].properties["intmode"]:
                intmode = True
            
        if intmode:
            bins = int(xmax-xmin+1)
            min = xmin - 0.5
            max = xmax + 0.5
        else:
            bins = properties["bins"]["value"]
            min = xmin
            max = xmax
        edges = details['range']
        if binlist and dictionary.has_key(binlist):
            edges = dictionary[binlist]
        edges = [float(edge) for edge in edges]
        if len(edges) == 2:
            edges = (min,max)
        info["min"] = min
        info["max"] = max
        info["bins"] = bins
        info["edges"] = edges
        variables.append((var,info))
    
    if len(variables) > 3:
        error("can't plot in more than 3 dimensions")
        return None

    variables.reverse()
    htemplate = getHistTemplate(variables)
    if not htemplate:
        error("could not create histogram template")
        return None
   
    ylabel = ""
    labels = []
    modifiedVarNames = []
    for index,(var,info) in enumerate(variables):
        if type(info["bins"]) is list:
            binWidth = -1.
        else:
            binWidth = float(info["max"] - info["min"]) / info["bins"]
        if index == 0:
            ylabel = properties["ylabel"]["value"]
        if info.has_key("fancy"):
            labels.append(info["fancy"])
        else:
            labels.append(var)
        modifiedVarNames.append(var)
        units = None
        if info.has_key("displayunits"):
            units = info["displayunits"]
            labels[-1] += " [%s]"%units
            if info.has_key("units"):
                scale = measure.convert(info["units"],units)
                modifiedVarNames[-1] += "*(%e)"%scale
        elif info.has_key("units"):
            units = info["units"]
            labels[-1] += " [%s]"%info["units"]
        if index==0:
            if binWidth > 0 and properties["showbinsize"]["value"]:
                ylabel += " / %s"% routines.round_to_n(binWidth,2)
                if units and properties["showunits"]["value"]:
                    ylabel += " %s"%units
    modifiedVarNames.reverse()
    expression = ":".join(modifiedVarNames)
    labels.append(ylabel)

    procRef = []
    histos = []
    stackedHistos = []
    stackedNames = []
    if stacked:
        stackedNames = stacked.split(',')
    
    for samplelist in samples:
        name = ",".join([s.name for s in samplelist]) 
        sample_props = samplelist[0].properties
        label = None
        if sample_props.has_key('label'):
            label = sample_props['label']
        else:
            all_labels_present = True
            for s in samplelist:
                if not s.title:
                    all_labels_present = False
                    break
            if all_labels_present:
                label = ",".join([s.title for s in samplelist])
        if not cuts and objects["cut"]:
            label += " "+str(cuts)
        localCuts = cuts
        if sample_props.has_key("cuts"):
            localCuts &= Cut(sample_props["cuts"])
        hist = htemplate.Clone()
        
        routines.draw_samples(samplelist, expression, hist, cuts = localCuts)

        if label:
            hist.SetTitle(label)
            hist.SetName(label)
        else:
            hist.SetTitle(name)
            hist.SetName(name)
        
        hist.SetLineWidth(properties["linewidth"]["value"])
        hist.SetMarkerSize(properties["markersize"]["value"])
        hist.intmode = intmode
        hist.decorate(**sample_props)
        if properties["normalize"]["value"].lower() == "unit":
            hist.norm = 1
        if reference:
            if name in reference:
                procRef.append(hist)
        if norm == name:
            normHist = hist
        if name in stackedNames:
            stackedHistos.append(hist)
        else:
            histos.append(hist)
    
    if len(procRef) > 0:
        procRef[0].visible = False
        integral = procRef[1].Integral()
        refHist = procRef[1].clone()
        refHist.Scale(1./integral)
        otherhist = procRef[0].clone()
        otherhist.Scale(1./procRef[0].Integral())
        difference = (otherhist - refHist)
        refGraph = refHist.toGraph()
        refGraph.setErrorsFromHist(difference)
        refGraph.Scale(integral)
        refGraph.visible = True
        refGraph.inlegend = False
        refGraph.format="a2"
        refGraph.colour = otherhist.colour
        refGraph.SetFillStyle(otherhist.GetFillStyle())
        refGraph.SetFillColor(otherhist.GetFillColor())
        refGraph.GetXaxis().SetLimits(variables[0][1]["min"],variables[0][1]["max"])
        refGraph.GetXaxis().SetRangeUser(variables[0][1]["min"],variables[0][1]["max"])
        histos = [refGraph] + histos
        
    #ROOT.gStyle.SetHatchesSpacing(properties["linewidth"]["value"])
    ROOT.gStyle.SetHatchesLineWidth(properties["linewidth"]["value"])
    ROOT.gStyle.SetTextSize(properties["textsize"]["value"])

    textlabel = None
    if properties["label"]["value"] != "" and objects["label"]:
        textlabel = routines.makeLabel(properties["labelx"]["value"],properties["labely"]["value"],properties["label"]["value"])

    canvas.Clear()
    routines.draw(
               histos,
               pad = canvas,
               title = properties["title"]["value"],
               showlegend = properties["showlegend"]["value"],
               axislabels = labels,
               textlabels = textlabel,
               yscale = properties["yscale"]["value"],
               xscale = properties["yscale"]["value"],
               greedylegend = properties["legendmode"]["value"],
               minimum = 0.)
    blankCanvas = False
    return histos

def draw(hists):
    
    if not hists:
        return
    if not type(hists) is list:
        hists = [hists]
    if len(hists) == 0:
        return
    axisLabels = hists[0].axisLabels
    ylabel = hists[0].ylabel
     
    canvas.Clear()
    
    #ROOT.gStyle.SetHatchesSpacing(properties["linewidth"]["value"])
    ROOT.gStyle.SetHatchesLineWidth(properties["linewidth"]["value"])
    ROOT.gStyle.SetTextSize(properties["textsize"]["value"])

    textlabel = None
    if properties["label"]["value"] != "":
        textlabel = routines.makeLabel(properties["labelx"]["value"],properties["labely"]["value"],properties["label"]["value"])
    
    drawHistos(canvas,hists,properties["title"]["value"],axisLabels,label=textlabel,ylabel=ylabel,normalized=properties["normalize"]["value"],
               showLegend=objects["legend"],
               yscale=properties["yscale"]["value"])

def func(expression,lower,upper):

    global canvas
    global blankCanvas
    f = ROOT.TF1("func",expression,float(lower),float(upper))
    if blankCanvas:
        drawObject(canvas,f)
    else:
        drawObject(canvas,f,"SAME")

def line(x1,y1,x2,y2):

    global canvas
    global blankCanvas
    l = ROOT.TLine(float(x1),float(y1),float(x2),float(y2))
    l.SetLineWidth(properties["linewidth"]["value"])
    if blankCanvas:
        drawObject(canvas,l)
    else:
        drawObject(canvas,l,"SAME")

def arrow(x1,y1,x2,y2,size,option):

    global canvas
    global blankCanvas
    global properties
    if size == None:
        a = ROOT.TArrow(float(x1),float(y1),float(x2),float(y2))
    elif option == None:
        a = ROOT.TArrow(float(x1),float(y1),float(x2),float(y2),float(size))
    else:
        a = ROOT.TArrow(float(x1),float(y1),float(x2),float(y2),float(size),option)
    a.SetLineWidth(properties["linewidth"]["value"])
    if blankCanvas:
        drawObject(canvas,a)
    else:
        drawObject(canvas,a,"SAME")

def graph(file):

    global canvas
    global blankCanvas
    global properties
    g = Graph(file=file)
    g.Stretch(1./1000.)
    g.Crop(0,100)
    g.SetLineWidth(properties["linewidth"]["value"])
    if blankCanvas:
        drawObject(canvas,g,"L")
    else:
        drawObject(canvas,g,"L SAME")

def getHistTemplate(variables):
    
    if len(variables) == 1:
        bins = variables[0][1]["bins"]
        if len(variables[0][1]["edges"]) == 2:
            args = (bins,) + tuple(variables[0][1]["edges"])
        else:
            args = variables[0][1]["edges"]
        return Hist(*args)
    elif len(variables) == 2:
        args = ()
        binsx = variables[0][1]["bins"]
        if len(variables[0][1]["edges"]) == 2:
            args += (binsx,) + tuple(variables[0][1]["edges"])
        else:
            args += tuple(variables[0][1]["edges"])
        binsy = variables[1][1]["bins"]
        if len(variables[1][1]["edges"]) == 2:
            args += (binsy,) + tuple(variables[1][1]["edges"])
        else:
            args += tuple(variables[1][1]["edges"])
        return Hist2D(*args)
    elif len(variables) == 3:
        args = ()
        binsx = variables[0][1]["bins"]
        if len(variables[0][1]["edges"]) == 2:
            args += (binsx,) + tuple(variables[0][1]["edges"])
        else:
            args += tuple(variables[0][1]["edges"])
        binsy = variables[1][1]["bins"]
        if len(variables[1][1]["edges"]) == 2:
            args += (binsy,) + tuple(variables[1][1]["edges"])
        else:
            args += tuple(variables[1][1]["edges"])
        binsz = variables[2][1]["bins"]
        if len(variables[2][1]["edges"]) == 2:
            args += (binsz,) + tuple(variables[2][1]["edges"])
        else:
            args += tuple(variables[2][1]["edges"])
        return Hist3D(*args)
    else:
        return None

def load(filename):
    
    manager.load(os.path.expandvars(filename))
    
def plug(filename):
    
    manager.plug(os.path.expandvars(filename))

def param(parameter,value=None):
    
    global properties
    global canvas
    if not properties.has_key(parameter):
        error("unknown property: %s"%parameter)
        return
    properties[parameter]["value"] = value
    canvas.Modified()
    canvas.Update()
    #reset(canvasOnly=True)

def mode(value):

    global plotMode
    plotMode = value
        
def show(object,value):
    
    global objects
    if object not in objects.keys():
        error("unknown object: %s"%object)
        return
    value = value.lower().capitalize()
    if value not in ["True","False"]:
        error("Invalid value: expected boolean (True/False)")
        return
    objects[object] = value == "True"

def showBranches():
    
    for variable in localVariableMeta.keys():
        print variable
        
def addVariable(variable, details=None):

    if not details:
        details = {"range":(0,1),
                   "type":"float"}
    if not localVariableMeta.has_key(variable):
        localVariableMeta[variable] = details

def setRange(variable,min,max):
    
    try:
        min = float(min)
        max = float(max)
    except:
        error("Could not parse max and/or min value")
        return
    if min >= max:
        error("Min >= Max")
        return
    if not localVariableMeta.has_key(variable):
        error("Variable %s is not yet in metadata. Plot this variable first to load the metadata.")
        return
    localVariableMeta[variable]["range"]=(min,max)

def setFancy(variable,fancy):

    if not localVariableMeta.has_key(variable):
        error("Variable %s is not yet in metadata. Plot this variable first to load the metadata.")
        return
    if fancy.upper() == "NONE":
        del localVariableMeta[variable]["fancy"]
    else:
        localVariableMeta[variable]["fancy"]=fancy

def setUnits(variable,units):

    if not localVariableMeta.has_key(variable):
        error("Variable %s is not yet in metadata. Plot this variable first to load the metadata.")
        return
    if units.upper() == "NONE":
        del localVariableMeta[variable]["units"]
    else:
        localVariableMeta[variable]["units"]=units

def setDisplayUnits(variable,displayunits):

    if not localVariableMeta.has_key(variable):
        error("Variable %s is not yet in metadata. Plot this variable first to load the metadata.")
        return
    if displayunits.upper() == "NONE":
        del localVariableMeta[variable]["displayunits"]
    else:
        localVariableMeta[variable]["displayunits"]=displayunits

def setType(variable,typename):

    if not localVariableMeta.has_key(variable):
        error("Variable %s is not yet in metadata. Plot this variable first to load the metadata.")
        return
    if typename.upper() == "NONE":
        del localVariableMeta[variable]["type"]
    else:
        localVariableMeta[variable]["type"]=typename.lower()
  
def save(filename):
    
    routines.save_pad(canvas,filename,properties["imageformat"]["value"])
    
def clear():
    
    global canvas
    global blankCanvas
    blankCanvas = True
    canvas.Clear()
    canvas.Modified()
    canvas.Update()

def reset(canvasOnly=False):
    
    global canvas
    global variableRangeMap
    global blankCanvas
    blankCanvas = True
    canvas = Canvas(uuid.uuid4().hex,uuid.uuid4().hex,0,0,properties["canvaswidth"]["value"],properties["canvasheight"]["value"])
    if not canvasOnly:
        localVariableMeta = {}
