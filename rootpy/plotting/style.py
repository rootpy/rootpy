from ROOT import TStyle, TColor, TGaxis, gROOT
from matplotlib.colors import colorConverter

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

def convert_marker(marker, mode):
    """
    Convert *marker* to ROOT or matplotlib format.

    *marker* will be interpreted as a ROOT marker index if an integer.
    Otherwise, it will be saved as a matplotlib marker.  *mode* can be
    'mpl' to return a matplotlib value or 'root' to return a ROOT index.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    keys, values = markers.keys(), markers.values()
    try: # marker is a ROOT marker index
        marker = int(marker)
        if mode == 'root':
            return marker
        return keys[values.index(marker)]
    except: # marker is a matplotlib value
        if mode == 'root':
            return markers[marker]
        return marker

linestyles = { "solid":1,
               "dashed":2,
               "dotted":3,
               "dashdot":4,
               "longdashdot":5,
               "longdashdotdotdot":6,
               "longdash":7,
               "longdashdotdot":8,
               "verylongdash":9,
               "verylongdashdot":10}

def convert_linestyle(style, mode):
    """
    Convert *style* to ROOT or matplotlib format.

    *style* will be interpreted as a ROOT linestyle index if an integer.
    Otherwise, it will be saved as a matplotlib linestyle.  *mode* can be
    'mpl' to return a matplotlib value or 'root' to return a ROOT index.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    keys, values = linestyles.keys(), linestyles.values()
    try: # style is a ROOT style index
        style = int(style)
        if mode == 'root':
            return style
        else:
            return keys[values.index(style)]
    except: # style is a matplotlib value
        if mode == 'root':
            return linestyles[style]
        return style

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

def convert_fill(fill, mode):
    """
    Convert *fill* to ROOT or matplotlib format.

    *fill* will be interpreted as a ROOT fill index if an integer.  Otherwise,
    it will be saved as a matplotlib hatch value.  *mode* can be 'mpl' to
    return a matplotlib hatch value, 'root' to return a ROOT fill index, or
    one of the strings 'hollow' or 'solid', which are mapped to ROOT indices.

    See http://matplotlib.sourceforge.net/api/artist_api.html#matplotlib.patches.Patch.set_hatch
    See http://root.cern.ch/root/html/TAttFill.html
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    keys, values = fills.keys(), fills.values()
    try: # fill is a ROOT fill index
        fill = int(fill)
        if mode == 'root':
            return fill
        else:
            return keys[values.index(fill)]
    except: # fill is a matplotlib value
        if mode == 'root':
            return fills[fill]
        return fill

def convert_color(color, mode):
    """
    Convert *color* to a TColor if *mode='root'* or to (r,g,b,a) if 'mpl'.

    The *color* argument can be a ROOT TColor or color index, an *RGB*
    or *RGBA* sequence or a string in any of several forms:
    
        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a float, like '0.4', indicating gray on a 0-1 scale

    if *arg* is *RGBA*, the *A* will simply be discarded.

    >>> convert_color(2)
    (1.0, 0.0, 0.0)
    >>> convert_color('b')
    (0.0, 0.0, 1.0)
    >>> convert_color('blue')
    (0.0, 0.0, 1.0)
    >>> convert_color('0.25')
    (0.25, 0.25, 0.25)
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    # if color is None:
    #     return None
    # elif color == 'none' or color == 'None':
    #     return 'none'
    try: # color is a TColor
        color = TColor(color)
        rgb = color.GetRed(), color.GetGreen(), color.GetBlue()
        return convert_color(rgb, mode)
    except (TypeError, ReferenceError):
        pass
    try: # color is a ROOT color index
        color = gROOT.GetColor(color)
        rgb = color.GetRed(), color.GetGreen(), color.GetBlue()
        return convert_color(rgb, mode)
    except (TypeError, ReferenceError):
        pass
    try: # color is an (r,g,b) tuple from 0 to 255
        if max(color) > 1.:
            color = [x/255. for x in color][0:3]
    except TypeError:
        pass
    # color is something understood by matplotlib
    color = colorConverter.to_rgb(color)
    if mode == 'root':
        return TColor.GetColor(*color)
    return color


def getStyle(name="ATLAS"):

    style = None
    if name.upper() == "ATLAS":
        try:
            from atlastools.style import getstyle
            style = getstyle()
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
        style.SetLegendBorderSize(0)

    return style
