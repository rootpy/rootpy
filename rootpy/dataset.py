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

    class Fileset(namedtuple('FilesetBase', Dataset._fields + ('files', 'treename'))):

        def split(partitions):
            
            filesets = [Fileset._make(self) for i in xrange(partitions)]
            for fileset in filesets:
                fileset.files = []
            files = self.files[:]
            while len(files) > 0:
                for fileset in filesets:
                    if len(files) > 0:
                        fileset.files.append(files.pop(0))
                    else:
                        break
            return filesets

    return Fileset

@memoized
def Treeset():

    return namedtuple('Treeset', Dataset._fields + ('trees',))
