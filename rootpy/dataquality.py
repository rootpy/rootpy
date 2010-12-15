import xml.etree.ElementTree as ET
import copy
from operator import itemgetter
use_yaml = True
try:
    import yaml
except:
    use_yaml = False

class GRL(object):

    def __init__(self, grl = None):

        self.__grl = {}
        if type(grl) is dict:
            self.__grl = grl
        elif type(grl) in [str, file]:
            filename = grl
            if type(grl) is file:
                filename = grl.name
            if filename.endswith('.xml'):
                tree = ET.parse(grl)
                lbcols = tree.getroot().findall('NamedLumiRange/LumiBlockCollection')
                for lbcol in lbcols:
                    run = int(lbcol.find('Run').text)
                    lbs = lbcol.findall('LBRange')
                    for lb in lbs:
                        self.insert(run, (int(lb.attrib['Start']),int(lb.attrib['End'])))
            elif filename.endswith('.yml'):
                if use_yaml:
                    self.__grl = yaml.load(grl)
                else:
                    raise ImportError("PyYAML module not found")
            else:
                raise ValueError("File %s is not recognized as a valid GRL format"% filename)
            self.optimize()

    def __repr__(self):
        
        return self.__str__()

    def __str__(self):

        output = ""
        for run in sorted(self.__grl.iterkeys()):
            lbranges = self.__grl[run]
            output += "RUN: %i\n"%run
            output += "LUMIBLOCKS:\n"
            for lbrange in lbranges:
                output += "\t%i --> %i\n"% lbrange
        return output

    def __getitem__(self, index):

        return self.__grl[index]
    
    def __contains__(self, runlb):
        """
        Pass the tuple (run, lbn)
        """
        if self.__grl.has_key(runlb[0]):
            lbranges = self.__grl[runlb[0]]
            for lbrange in lbranges:
                if runlb[1] >= lbrange[0] and runlb[1] <= lbrange[1]:
                    return True
        return False

    def __iter__(self):

        return iter(self.__grl.items())

    def insert(self, run, lbrange):

        if self.__grl.has_key(run):
            self.__grl[run].append(lbrange)
        else:
            self.__grl[run] = [lbrange]
    
    def optimize(self):
        """
        Sort and merge lumiblock ranges
        """
        for run, lbranges in self:
            lbranges.sort(key=itemgetter(0))
            if len(lbranges) > 1:
                first = 0
                last = len(lbranges)-1
                while first != last:
                    next = first + 1
                    merged = False
                    while next <= last: 
                        if lbranges[first][1] >= lbranges[next][1]:
                            for index in range(first+1,next+1):
                                lbranges.pop(next)
                            merged = True
                            break
                        elif lbranges[first][1] >= lbranges[next][0]:
                            lbranges[first] = (lbranges[first][0],lbranges[next][1])
                            for index in range(first+1,next+1):
                                lbranges.pop(next)
                            merged = True
                            break
                        next += 1
                    last = len(lbranges)-1
                    if not merged:
                        first += 1
                    
    def __add__(self, other):

        grlcopy = copy.deepcopy(self)
        for run, lbranges in other:
            for lbrange in lbranges:
                grlcopy.insert(run, lbrange)
        self.optimize()
        return grlcopy

    def write(self, filename, format = 'xml'):

        if format == 'xml':
            root = ET.Element('LumiRangeCollection')
            subroot = ET.SubElement(root,'NamedLumiRange')
            for run in sorted(self.__grl.iterkeys()):
                lumiblocks = self.__grl[run]
                lbcol = ET.SubElement(subroot,'LumiBlockCollection')
                runelement = ET.SubElement(lbcol,'Run')
                runelement.text = str(run)
                for lumiblock in lumiblocks:
                    lbrange = ET.SubElement(lbcol,'LBRange')
                    lbrange.set("Start",str(lumiblock[0]))
                    lbrange.set("End",str(lumiblock[1]))
            tree = ET.ElementTree(root)
            tree.write(filename)
