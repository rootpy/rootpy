from rootpy.decorators import memoized
from collections import namedtuple

@memoized
def Namedset():

    return namedtuple('Namedset', 'name title meta properties')

@memoized
def Dataset():

    return namedtuple('Dataset', Namedset._fields + ('datatype', 'classtype', 'weight'))

@memoized
def Fileset():

    return namedtuple('Fileset', Dataset._fields + ('files', 'treename'))

@memoized
def Treeset():

    return namedtuple('Treeset', Dataset._fields + ('trees',))
