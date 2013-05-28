# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from collections import namedtuple


Namedset = namedtuple('Namedset', 'name title label tags meta properties')
Dataset = namedtuple('Dataset', Namedset._fields + ('datatype', 'classtype', 'weight'))


class Fileset(namedtuple('Fileset', Dataset._fields + ('files', 'treename'))):

    def split(self, partitions):

        files = self.files[:]
        fileset_files = [[] for _ in xrange(partitions)]
        while len(files) > 0:
            for fileset in fileset_files:
                if len(files) > 0:
                    fileset.append(files.pop(0))
                else:
                    break
        mydict = self._asdict()
        filesets = []
        for fileset in fileset_files:
            mydict['files'] = fileset
            filesets.append(Fileset(**mydict))
        return filesets


class Treeset(namedtuple('Treeset', Dataset._fields + ('trees',))):

    def GetEntries(self, *args, **kwargs):

        return sum([tree.GetEntries(*args, **kwargs) for tree in self.trees])

    def Scale(self, value):

        for tree in self.trees:
            tree.Scale(value)

    def __iter__(self):

        for tree in self.trees:
            yield tree

    def Draw(self, *args, **kwargs):

        for tree in self.trees:
            tree.Draw(*args, **kwargs)
