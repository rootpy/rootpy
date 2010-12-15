import xml.etree.ElementTree as ET
import copy
from operator import itemgetter

class GRL(object):

    def __init__(self, grl = None):

        self.grl = {}
        if type(grl) is dict:
            self.grl = grl
        elif type(grl) in [str, file]:
            tree = ET.parse(grl)
            lbcols = tree.getroot().findall('NamedLumiRange/LumiBlockCollection')
            for lbcol in lbcols:
                run = int(lbcol.find('Run').text)
                lbs = lbcol.findall('LBRange')
                for lb in lbs:
                    self.__insert(run, (int(lb.attrib['Start']),int(lb.attrib['End'])))
            self.__optimize()

    def __contains__(self, runlb):
        """
        Pass the tuple (run, lbn)
        """
        if self.grl.has_key(runlb[0]):
            lbranges = self.grl[runlb[0]]
            for lbrange in lbranges:
                if runlb[1] >= lbrange[0] and runlb[1] <= lbrange[1]:
                    return True
        return False

    def __iter__(self):

        return iter(self.grl.items())

    def __insert(self, run, lbrange):

        if self.grl.has_key(run):
            """ TODO
            curr_lbranges = self.grl[run]
            for curr_lbrange in curr_lbranges:
                # do they intersect?
                if len(set(range()) & set(range())) > 0:
            """
            self.grl[run].append(lbrange)
        else:
            self.grl[run] = [lbrange]
    
    def __optimize(self):
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
                    while next <= last: 
                        if lbranges[first][1] > lbranges[next][1]:
                            for index in range(first+1,next+1):
                                lbranges.pop(next)
                        elif lbranges[first][1] > lbranges[next][0]:
                            lbranges[first] = (lbranges[first][0],lbranges[next][1])
                            for index in range(first+1,next+1):
                                lbranges.pop(next)
                        next += 1
                    first += 1
                    last = len(lbranges)-1

    def __add__(self, other):

        grlcopy = self.__deepcopy__()
        for run, lbranges in other:
            for lbrange in lbranges:
                grlcopy.__insert(run, lbrange)
        self.__optimize()
        return grlcopy

    def write(self, filename):

        root = ET.Element('LumiRangeCollection')
        subroot = ET.SubElement(root,'NamedLumiRange')
        for run,lumiblocks in self.grl.items():
            lbcol = ET.SubElement(subroot,'LumiBlockCollection')
            runelement = ET.SubElement(lbcol,'Run')
            runelement.text = str(run)
            for lumiblock in lumiblocks:
                lbrange = ET.SubElement(lbcol,'LBRange')
                lbrange.set("Start",str(lumiblock[0]))
                lbrange.set("End",str(lumiblock[1]))
        tree = ET.ElementTree(root)
        tree.write(filename)
