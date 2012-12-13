# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
A module containing scale factors for physical quantities
"""

yotta = 1E24
zetta = 1E21
exa = 1E18
peta = 1E15
tera = 1E12
giga = 1E9
mega = 1E6
kilo = 1E3
hecto = 1E2
deka = 1E1
deci = 1E-1
centi = 1E-2
milli = 1E-3
micro = 1E-6
nano = 1E-9
pico = 1E-12
femto = 1E-15
atto = 1E-18
zepto = 1E-21
yocto = 1E-24

prefix = {
    'Y': yotta,
    'Z': zetta,
    'E': exa,
    'P': peta,
    'T': tera,
    'G': giga,
    'M': mega,
    'k': kilo,
    'h': hecto,
    'da': deka,
    'd': deci,
    'c': centi,
    'm': milli,
    '#mu': micro,
    'mu': micro,
    'n': nano,
    'p': pico,
    'f': femto,
    'a': atto,
    'z': zepto,
    'y': yocto
}


def convert(origin, target):
    """
    Return the factor required to multiply the
    origin by to express in terms of target
    """
    origin_scale = 1
    for key, value in prefix.items():
        if origin.startswith(key):
            origin_scale = value
            break
    target_scale = 1
    for key, value in prefix.items():
        if target.startswith(key):
            target_scale = value
            break
    return origin_scale / target_scale
