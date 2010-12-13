import elementtree.ElementTree as ET
import copy

class GRL(object):

    def __init__(grl=None):

        self.grl = {}
        if type(grl) is dict:
            self.grl = grl
        elif type(grl) in [str, file]:
            tree = ET.parse(grl)
            lbcols = tree.findall('LumiRangeCollection/NamedLumiRange/LumiBlockCollection')
            for lbcol in lbcols:
                run = int(lbcol.find('Run').text)
                lumiblocks = []
                lbs = lbcol.finall('LBRange')
                for lb in lbs:
                    lumiblocks.append((int(lb.attrib['Start']),int(lb.attrib['End'])))
                self.grl[run] = lumiblocks
    
    def __add__(self, other):

        grlcopy = copy.deepcopy(self.grl)
        grlcopy.update(other.grl)
        return GRL(grlcopy)

    def write(self, filename):

        root = ET.Element('LumiRangeCollection')
        subroot = ET.Element(root,'NamedLumiRange')
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
