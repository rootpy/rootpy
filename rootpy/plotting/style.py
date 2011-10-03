from ROOT import TStyle, TColor, TGaxis, gROOT

__use_matplotlib = True
try:
    from matplotlib.colors import colorConverter
except ImportError:
    __use_matplotlib = False

##############################
#### Markers #################

markerstyles_root2mpl = {
    1 : '.',
    2 : '+',
    3 : '*',
    4 : 'o',
    5 : 'x',
    20 : 'o',
    21 : 's',
    22 : '^',
    23 : 'v',
    24 : 'o',
    25 : 's',
    26 : '^',
    27 : 'd',
    28 : '+',
    29 : '*',
    30 : '*',
    31 : '*',
    32 : 'v',
    33 : 'D',
    34 : '+',
    }
for i in range(6, 20):
    markerstyles_root2mpl[i] = '.'

markerstyles_mpl2root = {
    '.' : 1,
    ',' : 1,
    'o' : 4,
    'v' : 23,
    '^' : 22,
    '<' : 23,
    '>' : 22,
    '1' : 23,
    '2' : 22,
    '3' : 23,
    '4' : 22,
    's' : 25,
    'p' : 25,
    '*' : 3,
    'h' : 25,
    'H' : 25,
    '+' : 2,
    'x' : 5,
    'D' : 33,
    'd' : 27,
    '|' : 2,
    '_' : 2,
    0 : 1, # TICKLEFT
    1 : 1, # TICKRIGHT
    2 : 1, # TICKUP
    3 : 1, # TICKDOWN
    4 : 1, # CARETLEFT
    5 : 1, # CARETRIGHT
    6 : 1, # CARETUP
    7 : 1, # CARETDOWN
    'None' : '.',
    ' ' : '.',
    '' : '.',
    }

markerstyles_text2root = {
    "smalldot" : 6,
    "mediumdot" : 7,
    "largedot" : 8,
    "dot" : 9,
    "circle" : 20,
    "square" : 21,
    "triangle" : 22,
    "triangleup" : 22,
    "triangledown" : 23,
    "opencircle" : 24,
    "opensquare" : 25,
    "opentriangle" : 26,
    "opendiamond" : 27,
    "diamond" : 33,
    "opencross" : 28,
    "cross" : 34,
    "openstar" : 29,
    "fullstar" : 30,
    "star" : 29,
    }

def convert_markerstyle(inputstyle, mode, inputmode=None):
    """
    Convert *inputstyle* to ROOT or matplotlib format.

    Output format is determined by *mode* ('root' or 'mpl').  The *inputstyle*
    may be a ROOT marker style, a matplotlib marker style, or a description
    such as 'star' or 'square'.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    if inputmode is None:
        if inputstyle in markerstyles_root2mpl:
            inputmode = 'root'
        elif inputstyle in markerstyles_mpl2root or '$' in str(inputstyle):
            inputmode = 'mpl'
        elif inputstyle in markerstyles_text2root:
            inputmode = 'root'
            inputstyle = markerstyles_text2root[inputstyle]
        else:
            raise ValueError("%s is not a recognized marker style!"
                             % inputstyle)
    if inputmode == 'root':
        if inputstyle not in markerstyles_root2mpl:
            raise ValueError("%s is not a recognized ROOT marker style!"
                             % inputstyle)
        if mode == 'root':
            return inputstyle
        return markerstyles_root2mpl[inputstyle]
    else:
        if '$' in str(inputstyle):
            if mode == 'root':
                return 1
            else:
                return inputstyle
        if inputstyle not in markerstyles_mpl2root:
            raise ValueError("%s is not a recognized matplotlib marker style!"
                             % inputstyle)
        if mode == 'mpl':
            return inputstyle
        return markerstyles_mpl2root[inputstyle]



##############################
#### Lines ###################

linestyles_root2mpl = {
    1 : 'solid',
    2 : 'dashed',
    3 : 'dotted',
    4 : 'dashdot',
    5 : 'dashdot',
    6 : 'dashdot',
    7 : 'dashed',
    8 : 'dashdot',
    9 : 'dashed',
    10 : 'dashdot',
    }

linestyles_mpl2root = {
    'solid' : 1,
    'dashed' : 2,
    'dotted' : 3,
    'dashdot' : 4,
    }

linestyles_text2root = {
    'solid' : 1,
    'dashed' : 2,
    'dotted' : 3,
    'dashdot' : 4,
    'longdashdot' : 5,
    'longdashdotdotdot' : 6,
    'longdash' : 7,
    'longdashdotdot' : 8,
    'verylongdash' : 9,
    'verylongdashdot' : 10
    }

def convert_linestyle(inputstyle, mode, inputmode=None):
    """
    Convert *inputstyle* to ROOT or matplotlib format.

    Output format is determined by *mode* ('root' or 'mpl').  The *inputstyle*
    may be a ROOT line style, a matplotlib line style, or a description
    such as 'solid' or 'dotted'.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    if inputmode is None:
        if inputstyle in linestyles_root2mpl:
            inputmode = 'root'
        elif inputstyle in linestyles_mpl2root:
            inputmode = 'mpl'
        elif inputstyle in linestyles_text2root:
            inputmode = 'root'
            inputstyle = linestyles_text2root[inputstyle]
        else:
            raise ValueError("%s is not a recognized line style!"
                             % inputstyle)
    if inputmode == 'root':
        if inputstyle not in linestyles_root2mpl:
            raise ValueError("%s is not a recognized ROOT line style!"
                             % inputstyle)
        if mode == 'root':
            return inputstyle
        return linestyles_root2mpl[inputstyle]
    else:
        if inputstyle not in linestyles_mpl2root:
            raise ValueError("%s is not a recognized matplotlib line style!"
                             % inputstyle)
        if mode == 'mpl':
            return inputstyle
        return linestyles_mpl2root[inputstyle]



##############################
#### Fills ###################

fillstyles_root2mpl = {
    0: None,
    1001: None,
    3003: '.',
    3004: '/',
    3005: '\\',
    3006: '|',
    3007: '-',
    3011: '*',
    3012: 'o',
    3013: 'x',
    3019: 'O',
    }

fillstyles_mpl2root = {}
for key, value in fillstyles_root2mpl.items():
    fillstyles_mpl2root[value] = key

fillstyles_text2root = {
    'hollow' : 0,
    'solid' : 1001,
    }

def convert_fillstyle(inputstyle, mode, inputmode=None):
    """
    Convert *inputstyle* to ROOT or matplotlib format.

    Output format is determined by *mode* ('root' or 'mpl').  The *inputstyle*
    may be a ROOT fill style, a matplotlib hatch style, or a description
    such as 'hollow' or 'solid'.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("%s is not an understood value for mode" % mode)
    if inputmode is None:
        try:
            # inputstyle is a ROOT linestyle
            inputstyle = int(inputstyle)
            inputmode = 'root'
        except (TypeError, ValueError):
            if inputstyle is None:
                inputmode = 'mpl'
            elif inputstyle in fillstyles_text2root:
                inputmode = 'root'
                inputstyle = fillstyles_text2root[inputstyle]
            elif inputstyle[0] in fillstyles_mpl2root:
                inputmode = 'mpl'
            else:
                raise ValueError("%s is not a recognized fill style!"
                                 % inputstyle)
    if inputmode == 'root':
        if mode == 'root':
            return inputstyle
        if inputstyle in fillstyles_root2mpl:
            return fillstyles_root2mpl[inputstyle]
        return None
    else:
        if inputstyle is not None and inputstyle[0] not in fillstyles_mpl2root:
            raise ValueError("%s is not a recognized matplotlib fill style!"
                             % inputstyle)
        if mode == 'mpl':
            return inputstyle
        if inputstyle is None:
            return fillstyles_mpl2root[inputstyle]
        return fillstyles_mpl2root[inputstyle[0]]

# internal colors in case matplotlib is not available
__colors = { "white":0,
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
    # temp fix. needs improvement.
    if __use_matplotlib:
        try: # color is a TColor
            _color = TColor(color)
            rgb = _color.GetRed(), _color.GetGreen(), _color.GetBlue()
            return convert_color(rgb, mode)
        except (TypeError, ReferenceError):
            pass
        try: # color is a ROOT color index
            _color = gROOT.GetColor(color)
            rgb = _color.GetRed(), _color.GetGreen(), _color.GetBlue()
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
    else: # fall back on internal conversion
        if color in __colors.values():
            return color
        if color in __colors:
            color = __colors[color]
        elif type(color) is int:
            return color
        else:
            raise ValueError("Color %s is not understood" % repr(color))
    return color
