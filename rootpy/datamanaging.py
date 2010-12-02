import ROOT
from ntuple import Cut
import uuid
import os
from array import array
from collections import namedtuple
import metadata
import re
import warnings

SAMPLE_REGEX = re.compile("^(?P<name>[^(]+)(?:\((?P<type>[^)]+)\))?$")

Sample = namedtuple('Sample', 'name datatype classtype trees meta properties')

class DataManager:
    
    def __init__(self, verbose = False):
        
        self.verbose = verbose
        self.coreData = None
        self.coreDataName = None
        self.pluggedData = None
        self.pluggedDataName = None
        self.friendFiles = {}
        self.scratchFileName = "%s.root"% uuid.uuid4().hex
        self.scratchFile = ROOT.TFile.Open(self.scratchFileName,"recreate")
        self.variables = None
        self.objects = None
        self.datasets = None
    
    def __del__(self):
        
        if self.scratchFile:
            self.scratchFile.Close()
            os.remove(self.scratchFileName)
        if self.coreData:
            self.coreData.Close()
        if self.pluggedData:
            self.pluggedData.Close()
        for file in self.friendFiles.values():
            file.Close()
    
    def load(self, filename):
        
        if self.verbose: print "loading %s"%filename
        data = ROOT.TFile.Open(filename)
        if data:
            if self.coreData:
                self.coreData.Close()
            self.coreData = data
            self.coreDataName = filename
            varmeta = data.Get("variables.yml")
            if varmeta:
                self.variables = metadata.load(varmeta.GetTitle())
            else:
                warnings.warn("no variable metadata found")
            datasetmeta = data.Get("datasets.yml")
            if datasetmeta:
                self.datasets = metadata.load(datasetmeta.GetTitle())
            else:
                warnings.warn("no datasets metadata found")
            objectmeta = data.Get("trees.yml")
            if objectmeta:
                self.objects = metadata.load(objectmeta.GetTitle())
            else:
                warnings.warn("no trees metadata found")
        else:
            print "Could not open %s"%filename
    
    def plug(self, filename):
       
        if not self.coreData:
            print "Cannot plug in supplementary data with no core data!"
            return
        if not filename:
            if self.pluggedData:
                self.pluggedData.Close()
            self.pluggedData = None
            self.pluggedDataName = None
        else:
            if self.verbose: print "plugging in %s"%filename
            data = ROOT.TFile.Open(filename)
            if data:
                if self.pluggedData:
                    self.pluggedData.Close()
                self.pluggedData = data
                self.pluggedDataName = filename
            else:
                print "Could not open %s"%filename
         
    def get_object_by_name(self,name):
        
        for file,filename in [(self.pluggedData,self.pluggedDataName), (self.coreData,self.coreDataName)]:
            if file:
                object = file.Get(name)
                if object:
                    return (object,filename)
        return (None,None)
             
    def normalize_weights(self,trees,norm=1.):
        
        totalWeight = 0.
        for tree in trees:
            totalWeight += tree.GetWeight()
        for tree in trees:
            tree.SetWeight(norm*tree.GetWeight()/totalWeight)
    
    def get_tree(self, treepath, maxEntries=-1, fraction=-1, cuts=None):
        
        if cuts == None:
            cuts = Cut("")
        if self.verbose: print "Fetching tree %s..."%treepath
        inFile = None
        filename = ""
        tmpTree = None
        tree,filename = self.get_object_by_name(treepath)
        if not tree:
            if self.verbose: print "Tree %s not found!"%treepath
            return None
        friends = tree.GetListOfFriends()
        if friends:
            if len(friends) > 0 and self.verbose:
                print "Warning! ROOT does not play nice with friends where cuts are involved!"
            if len(friends) == 1:
                if self.verbose:
                    print "Since this tree has one friend, I will assume that it's friend is the core data (read-only) and you want that tree instead"
                    print "where cuts may refer to branches in this tree"
                tmpTree = tree
                friendTreeName = friends[0].GetTreeName()
                if self.friendFiles.has_key(friends[0].GetTitle()):
                    friendTreeFile = self.friendFiles[friends[0].GetTitle()]
                else:
                    friendTreeFile = ROOT.TFile.Open(friends[0].GetTitle())
                    self.friendFiles[friends[0].GetTitle()] = friendTreeFile
                filename = friends[0].GetTitle()
                tree = friendTreeFile.Get(friendTreeName)
                tree.AddFriend(tmpTree)
            elif len(friends) > 1 and self.verbose:
                print "Warning! This tree has multiple friends!"
        if not cuts.empty():
            print "Applying cuts %s"%cuts
            if friends:
                if len(friends) > 1 and self.verbose:
                    print "Warning: applying cuts on tree with multiple friends is not safely implemented yet"
            self.scratchFile.cd()
            tree = tree.CopyTree(str(cuts))
        originalNumEntries = tree.GetEntries()
        if fraction > -1.:
            entries = tree.GetEntries()
            if self.verbose: print "Extracting %.1f%% of the tree which contains %i entries."% (fraction*100., entries)
            newEntries = int(fraction*entries)
            self.scratchFile.cd()
            tree = tree.CloneTree(newEntries)
        elif maxEntries > -1 and tree.GetEntries() > maxEntries:
            if self.verbose:
                print "Number of entries in tree exceeds maximum allowed by user: %i"% maxEntries
                print "Extracting %i of %i total entries"% (maxEntries, tree.GetEntries())
            self.scratchFile.cd()
            tree = tree.CloneTree(maxEntries)
        finalNumEntries = tree.GetEntries()
        if finalNumEntries > 0 and originalNumEntries != finalNumEntries:
            tree.SetWeight(tree.GetWeight()*float(originalNumEntries)/float(finalNumEntries))
        if self.verbose: print "Found %s with %i entries and weight %e"%(treepath, tree.GetEntries(), tree.GetWeight())
        if cuts:
            tree.SetName("%s:%s"% (tree.GetName(), cuts))
        return tree
    
    def get_sample(self, samplestring, treetype=None, cuts=None, maxEntries=-1, fraction=-1):
       
        if self.datasets is None or self.objects is None or self.variables is None:
            return None
        properties = {}
        if '|' in samplestring:
            samplestring,proptertystrings = samplestring.split('|')
            properties = dict([(property.split('=')) for property in proptertystrings.split(':')])
        else:
            formatstring = None
        samplestrings = samplestring.split('+')
        samples = []
        for string in samplestrings:
            sample_match = re.match(SAMPLE_REGEX, string)
            if not sample_match:
                raise SyntaxError("%s is not valid sample syntax"% string)
            samplename = sample_match.group('name')
            sampletype = sample_match.group('type')
            if sampletype is None and treetype is None:
                if self.objects.has_key('default'):
                    sampletype = self.objects['default']
                elif len(self.objects) is 1:
                    sampletype = self.objects.values()[0]
                else:
                    raise ValueError("No sample type specified yet no default exists")
            elif (treetype is not None) and (sampletype is not None) and (sampletype != treetype):
                raise ValueError("Conflicting sample types specified: %s and %s"% (sampletype, treetype))
            elif sampletype is None and treetype is not None:
                sampletype = treetype
            if sampletype not in self.objects.keys() and sampletype != 'default':
                raise ValueError("sample type %s is not defined"% sampletype)
            elif sampletype == 'default':
                raise ValueError("sample type cannot be 'default'")
            tree_paths, datatype, classtype = metadata.find_sample(samplename, sampletype, self.datasets, self.objects)
            trees = []
            for treepath in tree_paths:
                if self.verbose: print "==========================================================="
                trees.append(self.get_tree(treepath, maxEntries=maxEntries, fraction=fraction, cuts=cuts))
            for tree in trees:
                if tree is None:
                    raise RuntimeError("sample %s was not found"% samplename)
                # set aliases
                for branch in self.objects[sampletype]:
                    if tree.GetBranch(branch):
                        if self.variables.has_key(branch):
                            if self.variables[branch].has_key('alias'):
                                tree.SetAlias(self.variables[branch]['alias'],branch)
                        else:
                            warnings.warn("branch listed for tree type %s is not listed in variables.yml"% sampletype, RuntimeWarning)
                    #else:
                    #    warnings.warn("branch %s does not exist in tree %s"% (branch, tree.GetName()), RuntimeWarning)
            samples.append(Sample(samplename, datatype, classtype, trees, self.variables, properties))
        return samples
