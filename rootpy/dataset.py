from collections import namedtuple

Namedset = namedtuple('Namedset', 'name title meta properties')

Dataset = namedtuple('Dataset', Namedset._fields + ('datatype', 'classtype', 'weight'))

class Fileset(namedtuple('Fileset', Dataset._fields + ('files', 'treename'))):

    def split(self, partitions):
        
        files = self.files[:]
        fileset_files = [[] for i in xrange(partitions)]
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

Treeset = namedtuple('Treeset', Dataset._fields + ('trees',))
