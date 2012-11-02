import ROOT


class _StyleContainer(object):
    """
    Base class for grouping together an input style with ROOT and matplotlib
    styles.
    """
    def __init__(self, value, function):
        self._input = value
        self._root = function(value, 'root')
        self._mpl = function(value, 'mpl')

    def __call__(self, output_type=None):
        if not output_type:
            output_type = 'input'
        return getattr(self, '_' + output_type)

    def __repr__(self):
        return str(self._input)


##############################
#### Markers #################

markerstyles_root2mpl = {
    1: '.',
    2: '+',
    3: '*',
    4: 'o',
    5: 'x',
    20: 'o',
    21: 's',
    22: '^',
    23: 'v',
    24: 'o',
    25: 's',
    26: '^',
    27: 'd',
    28: '+',
    29: '*',
    30: '*',
    31: '*',
    32: 'v',
    33: 'D',
    34: '+',
    }
for i in range(6, 20):
    markerstyles_root2mpl[i] = '.'

markerstyles_mpl2root = {
    '.': 1,
    ',': 1,
    'o': 4,
    'v': 23,
    '^': 22,
    '<': 23,
    '>': 22,
    '1': 23,
    '2': 22,
    '3': 23,
    '4': 22,
    's': 25,
    'p': 25,
    '*': 3,
    'h': 25,
    'H': 25,
    '+': 2,
    'x': 5,
    'D': 33,
    'd': 27,
    '|': 2,
    '_': 2,
    0: 1,  # TICKLEFT
    1: 1,  # TICKRIGHT
    2: 1,  # TICKUP
    3: 1,  # TICKDOWN
    4: 1,  # CARETLEFT
    5: 1,  # CARETRIGHT
    6: 1,  # CARETUP
    7: 1,  # CARETDOWN
    'None': '.',
    ' ': '.',
    '': '.',
    }

markerstyles_text2root = {
    "smalldot": 6,
    "mediumdot": 7,
    "largedot": 8,
    "dot": 9,
    "circle": 20,
    "square": 21,
    "triangle": 22,
    "triangleup": 22,
    "triangledown": 23,
    "opencircle": 24,
    "opensquare": 25,
    "opentriangle": 26,
    "opendiamond": 27,
    "diamond": 33,
    "opencross": 28,
    "cross": 34,
    "openstar": 29,
    "fullstar": 30,
    "star": 29,
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


class MarkerStyle(_StyleContainer):
    """
    Container for grouping together ROOT and matplotlib marker styles.

    The *style* argument to the constructor may be a ROOT marker style,
    a matplotlib marker style, or one of the following descriptions:
    """
    __doc__ = __doc__[:__doc__.rfind('\n') + 1]
    __doc__ += '\n'.join(["    '%s'" % x
                          for x in markerstyles_text2root])
    del x
    __doc__ += """

    Examples:

    >>> style = MarkerStyle('opentriangle')
    >>> style('root')
    26
    >>> style('mpl')
    '^'
    """

    def __init__(self, style):
        _StyleContainer.__init__(self, style, convert_markerstyle)


##############################
#### Lines ###################

linestyles_root2mpl = {
    1: 'solid',
    2: 'dashed',
    3: 'dotted',
    4: 'dashdot',
    5: 'dashdot',
    6: 'dashdot',
    7: 'dashed',
    8: 'dashdot',
    9: 'dashed',
    10: 'dashdot',
    }

linestyles_mpl2root = {
    'solid': 1,
    'dashed': 2,
    'dotted': 3,
    'dashdot': 4,
    }

linestyles_text2root = {
    'solid': 1,
    'dashed': 2,
    'dotted': 3,
    'dashdot': 4,
    'longdashdot': 5,
    'longdashdotdotdot': 6,
    'longdash': 7,
    'longdashdotdot': 8,
    'verylongdash': 9,
    'verylongdashdot': 10
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
    try:
        inputstyle = int(inputstyle)
        if inputstyle < 1:
            inputstyle = 1
    except ValueError:
        pass
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


class LineStyle(_StyleContainer):
    """
    Container for grouping together ROOT and matplotlib line styles.

    The *style* argument to the constructor may be a ROOT line style,
    a matplotlib line style, or one of the following descriptions:
    """
    __doc__ = __doc__[:__doc__.rfind('\n') + 1]
    __doc__ += '\n'.join(["    '%s'" % x
                          for x in linestyles_text2root])
    del x
    __doc__ += """

    Examples:

    >>> style = LineStyle('verylongdashdot')
    >>> style('root')
    10
    >>> style('mpl')
    'dashdot'
    """

    def __init__(self, style):
        _StyleContainer.__init__(self, style, convert_linestyle)


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
    'hollow': 0,
    'solid': 1001,
    }


def convert_fillstyle(inputstyle, mode, inputmode=None):
    """
    Convert *inputstyle* to ROOT or matplotlib format.

    Output format is determined by *mode* ('root' or 'mpl').  The *inputstyle*
    may be a ROOT fill style, a matplotlib hatch style, 'hollow', or 'solid'.
    """
    mode = mode.lower()
    if mode != 'mpl' and mode != 'root':
        raise ValueError("'%s' is not a valid mode" % mode)
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
                raise ValueError("'%s' is not a valid fill style!"
                                 % inputstyle)
    if inputmode == 'root':
        if mode == 'root':
            return inputstyle
        if inputstyle in fillstyles_root2mpl:
            return fillstyles_root2mpl[inputstyle]
        raise ValueError("'%s' is not a valid fill style!" % inputstyle)
    else:
        if inputstyle is not None and inputstyle[0] not in fillstyles_mpl2root:
            raise ValueError("'%s' is not a valid matplotlib fill style!"
                             % inputstyle)
        if mode == 'mpl':
            return inputstyle
        if inputstyle is None:
            return fillstyles_mpl2root[inputstyle]
        return fillstyles_mpl2root[inputstyle[0]]


class FillStyle(_StyleContainer):
    """
    Container for grouping together ROOT and matplotlib fill styles.

    The *style* argument to the constructor may be a ROOT fill style,
    a matplotlib fill style, or one of the following descriptions:
    """
    __doc__ = __doc__[:__doc__.rfind('\n') + 1]
    __doc__ += '\n'.join(["    '%s'" % x
                          for x in fillstyles_text2root])
    del x
    __doc__ += """

    For an input value of 'solid', the matplotlib hatch value will be set to None,
    which is the same value as for 'hollow'.  The root2matplotlib functions will
    all check the ROOT value to see whether to make the fill solid or hollow.

    Examples:

    >>> style = FillStyle('hollow')
    >>> style('root')
    0
    >>> print style('mpl')
    None
    """

    def __init__(self, style):
        _StyleContainer.__init__(self, style, convert_fillstyle)


##############################
#### Colors ##################

_cnames = {
    'r'                    : '#FF0000', #@IgnorePep8
    'g'                    : '#00FF00',
    'b'                    : '#0000FF',
    'c'                    : '#00BFBF',
    'm'                    : '#BF00BF',
    'y'                    : '#BFBF00',
    'k'                    : '#000000',
    'w'                    : '#FFFFFF',
    'aliceblue'            : '#F0F8FF',
    'antiquewhite'         : '#FAEBD7',
    'aqua'                 : '#00FFFF',
    'aquamarine'           : '#7FFFD4',
    'azure'                : '#F0FFFF',
    'beige'                : '#F5F5DC',
    'bisque'               : '#FFE4C4',
    'black'                : '#000000',
    'blanchedalmond'       : '#FFEBCD',
    'blue'                 : '#0000FF',
    'blueviolet'           : '#8A2BE2',
    'brown'                : '#A52A2A',
    'burlywood'            : '#DEB887',
    'cadetblue'            : '#5F9EA0',
    'chartreuse'           : '#7FFF00',
    'chocolate'            : '#D2691E',
    'coral'                : '#FF7F50',
    'cornflowerblue'       : '#6495ED',
    'cornsilk'             : '#FFF8DC',
    'crimson'              : '#DC143C',
    'cyan'                 : '#00FFFF',
    'darkblue'             : '#00008B',
    'darkcyan'             : '#008B8B',
    'darkgoldenrod'        : '#B8860B',
    'darkgray'             : '#A9A9A9',
    'darkgreen'            : '#006400',
    'darkkhaki'            : '#BDB76B',
    'darkmagenta'          : '#8B008B',
    'darkolivegreen'       : '#556B2F',
    'darkorange'           : '#FF8C00',
    'darkorchid'           : '#9932CC',
    'darkred'              : '#8B0000',
    'darksalmon'           : '#E9967A',
    'darkseagreen'         : '#8FBC8F',
    'darkslateblue'        : '#483D8B',
    'darkslategray'        : '#2F4F4F',
    'darkturquoise'        : '#00CED1',
    'darkviolet'           : '#9400D3',
    'deeppink'             : '#FF1493',
    'deepskyblue'          : '#00BFFF',
    'dimgray'              : '#696969',
    'dodgerblue'           : '#1E90FF',
    'firebrick'            : '#B22222',
    'floralwhite'          : '#FFFAF0',
    'forestgreen'          : '#228B22',
    'fuchsia'              : '#FF00FF',
    'gainsboro'            : '#DCDCDC',
    'ghostwhite'           : '#F8F8FF',
    'gold'                 : '#FFD700',
    'goldenrod'            : '#DAA520',
    'gray'                 : '#808080',
    'green'                : '#008000',
    'greenyellow'          : '#ADFF2F',
    'honeydew'             : '#F0FFF0',
    'hotpink'              : '#FF69B4',
    'indianred'            : '#CD5C5C',
    'indigo'               : '#4B0082',
    'ivory'                : '#FFFFF0',
    'khaki'                : '#F0E68C',
    'lavender'             : '#E6E6FA',
    'lavenderblush'        : '#FFF0F5',
    'lawngreen'            : '#7CFC00',
    'lemonchiffon'         : '#FFFACD',
    'lightblue'            : '#ADD8E6',
    'lightcoral'           : '#F08080',
    'lightcyan'            : '#E0FFFF',
    'lightgoldenrodyellow' : '#FAFAD2',
    'lightgreen'           : '#90EE90',
    'lightgrey'            : '#D3D3D3',
    'lightpink'            : '#FFB6C1',
    'lightsalmon'          : '#FFA07A',
    'lightseagreen'        : '#20B2AA',
    'lightskyblue'         : '#87CEFA',
    'lightslategray'       : '#778899',
    'lightsteelblue'       : '#B0C4DE',
    'lightyellow'          : '#FFFFE0',
    'lime'                 : '#00FF00',
    'limegreen'            : '#32CD32',
    'linen'                : '#FAF0E6',
    'magenta'              : '#FF00FF',
    'maroon'               : '#800000',
    'mediumaquamarine'     : '#66CDAA',
    'mediumblue'           : '#0000CD',
    'mediumorchid'         : '#BA55D3',
    'mediumpurple'         : '#9370DB',
    'mediumseagreen'       : '#3CB371',
    'mediumslateblue'      : '#7B68EE',
    'mediumspringgreen'    : '#00FA9A',
    'mediumturquoise'      : '#48D1CC',
    'mediumvioletred'      : '#C71585',
    'midnightblue'         : '#191970',
    'mintcream'            : '#F5FFFA',
    'mistyrose'            : '#FFE4E1',
    'moccasin'             : '#FFE4B5',
    'navajowhite'          : '#FFDEAD',
    'navy'                 : '#000080',
    'oldlace'              : '#FDF5E6',
    'olive'                : '#808000',
    'olivedrab'            : '#6B8E23',
    'orange'               : '#FFA500',
    'orangered'            : '#FF4500',
    'orchid'               : '#DA70D6',
    'palegoldenrod'        : '#EEE8AA',
    'palegreen'            : '#98FB98',
    'palevioletred'        : '#AFEEEE',
    'papayawhip'           : '#FFEFD5',
    'peachpuff'            : '#FFDAB9',
    'peru'                 : '#CD853F',
    'pink'                 : '#FFC0CB',
    'plum'                 : '#DDA0DD',
    'powderblue'           : '#B0E0E6',
    'purple'               : '#800080',
    'red'                  : '#FF0000',
    'rosybrown'            : '#BC8F8F',
    'royalblue'            : '#4169E1',
    'saddlebrown'          : '#8B4513',
    'salmon'               : '#FA8072',
    'sandybrown'           : '#FAA460',
    'seagreen'             : '#2E8B57',
    'seashell'             : '#FFF5EE',
    'sienna'               : '#A0522D',
    'silver'               : '#C0C0C0',
    'skyblue'              : '#87CEEB',
    'slateblue'            : '#6A5ACD',
    'slategray'            : '#708090',
    'snow'                 : '#FFFAFA',
    'springgreen'          : '#00FF7F',
    'steelblue'            : '#4682B4',
    'tan'                  : '#D2B48C',
    'teal'                 : '#008080',
    'thistle'              : '#D8BFD8',
    'tomato'               : '#FF6347',
    'turquoise'            : '#40E0D0',
    'violet'               : '#EE82EE',
    'wheat'                : '#F5DEB3',
    'white'                : '#FFFFFF',
    'whitesmoke'           : '#F5F5F5',
    'yellow'               : '#FFFF00',
    'yellowgreen'          : '#9ACD32',
    }


def convert_color(color, mode):
    """
    Convert *color* to a TColor if *mode='root'* or to (r,g,b) if 'mpl'.

    The *color* argument can be a ROOT TColor or color index, an *RGB*
    or *RGBA* sequence or a string in any of several forms:

        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a float, like '0.4', indicating gray on a 0-1 scale

    if *arg* is *RGBA*, the transparency value will be ignored.
    """
    mode = mode.lower()
    if mode not in ['mpl', 'root']:
        raise ValueError("%s is not an understood value for mode" % mode)
    try:
        # color is an r,g,b tuple
        color = tuple([float(x) for x in color[:3]])
        if max(color) > 1.:
            color = tuple([x / 255. for x in color])
        if mode == 'root':
            return ROOT.TColor.GetColor(*color)
        return color
    except (ValueError, TypeError):
        pass
    if isinstance(color, basestring):
        if color in _cnames:
            # color is a matplotlib letter or an html color name
            color = _cnames[color]
        if color[0] == '#':
            # color is a hex value
            color = color.lstrip('#')
            lv = len(color)
            color = tuple(int(color[i:i + lv / 3], 16) for i in range(0, lv, lv / 3))
            if lv == 3:
                color = tuple(x * 16 + x for x in color)
            return convert_color(color, mode)
        # color is a shade of gray, i.e. '0.3'
        return convert_color((color, color, color), mode)
    try:
        # color is a TColor
        color = ROOT.TColor(color)
        color = color.GetRed(), color.GetGreen(), color.GetBlue()
        return convert_color(color, mode)
    except (TypeError, ReferenceError):
        pass
    try:
        # color is a ROOT color index
        if color < 1:
            color = 1
        color = ROOT.gROOT.GetColor(color)
        # Protect against the case a histogram with a custom color
        # is saved in a ROOT file
        if not color:
            # Just return black
            color = ROOT.gROOT.GetColor(1)
        color = color.GetRed(), color.GetGreen(), color.GetBlue()
        return convert_color(color, mode)
    except (TypeError, ReferenceError):
        pass
    raise ValueError("'%s' is not a valid color" % str(color))


class Color(_StyleContainer):
    """
    Container for grouping together ROOT and matplotlib colors.

    The *color* argument to the constructor can be a ROOT TColor or color index.
    If matplotlib is available, it can also accept an *RGB* or *RGBA* sequence,
    or a string in any of several forms:

        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a float, like '0.4', indicating gray on a 0-1 scale

    if *color* is *RGBA*, the *A* will simply be discarded.

    Examples:

    >>> color = Color(2)
    >>> color()
    2
    >>> color('mpl')
    (1.0, 0.0, 0.0)
    >>> color = Color('blue')
    >>> color('root')
    4
    >>> color('mpl')
    (0.0, 0.0, 1.0)
    >>> color = Color('0.25')
    >>> color('mpl')
    (0.25, 0.25, 0.25)
    >>> color('root')
    924
    """
    def __init__(self, color):
        _StyleContainer.__init__(self, color, convert_color)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
