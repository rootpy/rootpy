from ROOT import TStyle, TGaxis, gROOT

markers = { "":1,
            ".":1,
            "+":2,
            "*":3,
            "o":4,
            "x":5,
            "smalldot":6,
            "mediumdot":7,
            "largedot":8,
            "dot":9,
            "circle":20,
            "square":21,
            "triangle":22,
            "triangleup":22,
            "triangledown":23,
            "opencircle":24,
            "opensquare":25,
            "opentriangle":26,
            "diamond":27,
            "cross":28,
            "openstar":29,
            "fullstar":30,
            "star":30}

colours = { "":1,
            "white":0,
            "black":1,
            "red":2,
            "green":3,
            "blue":4,
            "yellow":5,
            "purple":6,
            "aqua":7,
            "forest":8,
            "violet":9}

lines = { "":1,
          "-":2,
          ".":3,
          "-.":4,
          "_.":5,
          "_...":6,
          "_":7,
          "_..":8,
          "__":9,
          "__.":10}

def getStyle(name="MINE"):

    style = None
    if name.upper() == "MINE":
        style = myStyle()
    elif name.upper() == "ATLAS":
        try:
            from atlasstyle import AtlasStyle
            style = AtlasStyle.AtlasStyle()
        except:
            print "You need to put the atlasstyle module in your ROOT macro path"
    
    if style != None:
        #style.SetTitleH(0.08)
        #style.SetTitleW(1.)
        style.SetOptTitle(1)
        style.SetTitleX(0.5)
        style.SetTitleAlign(23)
        style.SetTitleColor(1)
        style.SetTitleFillColor(0)
        style.SetTitleStyle(0)
        style.SetTitleBorderSize(0)
        style.SetTitleFontSize(0.07)

    return style

def myStyle():

    style = TStyle("MINE","My Style")
    icol=0
    font=43
    style.SetPalette(1)
        
    #style.SetTextSize(0.1)
    
    style.SetPadTopMargin(0.1)
    style.SetPadLeftMargin(0.15)
    style.SetPadRightMargin(0.1)
    style.SetPadBottomMargin(0.15)
    
    style.SetFrameBorderMode(icol)
    style.SetCanvasBorderMode(icol)
    style.SetPadBorderMode(icol)
    style.SetPadColor(icol)
    style.SetCanvasColor(icol)
    style.SetLegendBorderSize(0)
    
    style.SetLabelFont(font,"XYZ")
    style.SetTitleFont(font,"XYZ")
    style.SetLabelSize(16,"XYZ")
    style.SetTitleSize(20,"XYZ")
    style.SetLabelOffset(0.006,"XYZ")
    style.SetTitleOffset(1.1,"XYZ")
    #do not display any of the standard histogram decorations
    style.SetOptStat(0)
    style.SetOptFit(0)
    # put tick marks on top and RHS of plots
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetLineWidth(1)

    style.SetHatchesSpacing(1.0)
    style.SetHatchesLineWidth(1)
    return style
