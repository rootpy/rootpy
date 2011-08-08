from ROOT import TStyle, TGaxis, gROOT

markers = { ".":1,
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
            "opendiamond":27,
            "diamond":33,
            "opencross":28,
            "cross":34,
            "openstar":29,
            "openstar":30,
            "star":29}

colors = { "white":0,
           "black":1,
           "red":2,
           "dullred":46,
           "green":3,
           "dullgreen":30,
           "blue":4,
           "dullblue":38,
           "yellow":5,
           "purple":6,
           "aqua":7,
           "forest":8,
           "violet":9}

lines = { "solid":1,
          "dashed":2,
          "dotted":3,
          "dashdot":4,
          "longdashdot":5,
          "longdashdotdotdot":6,
          "longdash":7,
          "longdashdotdot":8,
          "verylongdash":9,
          "verylongdashdot":10}

fills = { "hollow":0,
          "solid":1001,
          ".": 3003,
          "*": 3011,
          "o": 3012,
          "O": 3019,
          "x": 3013,
          '\\': 3005,
          '/': 3004,
          '|': 3006,
          '-': 3007}

def getStyle(name="ATLAS"):

    style = None
    if name.upper() == "ATLAS":
        try:
            from atlastools.style import getstyle
            style = getstyle()
        except:
            print "You need to put the atlasstyle module in your ROOT macro path"
    
    """ 
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
        style.SetLegendBorderSize(0)
    """
    return style
