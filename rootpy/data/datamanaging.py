from ..tree import Tree
from ..io import File, open as ropen
from ..tree import Cut
from .dataset import Treeset
import uuid
import os
from array import array
import metadata
import re
import warnings
import ROOT

SAMPLE_REGEX = re.compile("^(?P<name>[^(]+)(?:\((?P<type>[^)]+)\))?$")

class DataManager(object):
    
    def __init__(self, verbose = False):
        
        self.verbose = verbose
        self.coreData = None
        self.coreDataName = None
        self.pluggedData = None
        self.pluggedDataName = None
        self.friendFiles = {}
        self.scratchFileName = "%s.root"% uuid.uuid4().hex
        self.scratchFile = File(self.scratchFileName,"recreate")
        ROOT.gROOT.GetListOfFiles().Remove(self.scratchFile)
        self.variables = None
        self.trees = None
        self.datasets = None
        self.__use_rootfs = True
        self.root = None
        self.files = {}
    
    def __del__(self):
        
        self.scratchFile.Close()
        if os:
            os.remove(self.scratchFileName)
        if self.coreData:
            self.coreData.Close()
        if self.pluggedData:
            self.pluggedData.Close()
        for file in self.friendFiles.values():
            file.Close()
        for file in self.files.values():
            if file:
                file.Close()
    
    def load(self, filename, metadir=None):
        
        filename = os.path.expandvars(filename)
        if os.path.isdir(filename):
            self.__use_rootfs = False
        elif os.path.isfile(filename):
            self.__use_rootfs = True
        else:
            print "%s does not exist"% filename
            return False
        if self.__use_rootfs:
            if self.verbose: print "loading %s"%filename
            data = ropen(filename)
            if not data:
                print "Could not open %s"% filename
                return False
            if self.coreData:
                self.coreData.Close()
            self.coreData = data
            self.coreDataName = filename
        else:
            self.root = filename
            if self.coreData:
                self.coreData.Close()
        dataroot = os.path.dirname(filename)
        # get metadata
        for meta in ["variables", "datasets", "trees"]:
            metafile = "%s.yml"% meta
            if metadir:
                metafile_user = os.path.join(metadir, metafile)
                if os.path.isfile(metafile_user):
                    print "loading %s"% metafile_user
                    setattr(self,meta,metadata.load(metafile_user))
                    continue
            else:    
                if os.path.isfile(metafile):
                    print "loading %s"% metafile
                    setattr(self,meta,metadata.load(metafile))
                    continue
                metafile_data = os.path.join(dataroot, metafile)
                if os.path.isfile(metafile_data):
                    print "loading %s"% metafile_data
                    setattr(self,meta,metadata.load(metafile_data))
                    continue
                if os.environ.has_key('DATAROOT'):
                    dataroot_central = os.environ['DATAROOT']
                    metafile_central = os.path.join(dataroot_central, metafile)
                    if os.path.isfile(metafile_central):
                        print "loading %s"% metafile_central
                        setattr(self,meta,metadata.load(metafile_central))
                        continue
            print "Could not find %s.yml in $DATAROOT, %s or current working directory"% (meta, dataroot)
            return False
        return True

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
            data = File(filename)
            if data:
                if self.pluggedData:
                    self.pluggedData.Close()
                self.pluggedData = data
                self.pluggedDataName = filename
            else:
                print "Could not open %s"%filename
         
    def get_object_by_name(self,name):
        
        if self.__use_rootfs:
            for file,filename in [(self.pluggedData,self.pluggedDataName), (self.coreData,self.coreDataName)]:
                if file:
                    object = file.Get(name)
                    if object:
                        return (object,filename)
        else:
            path = name.split('/')
            filename = os.path.normpath(os.path.join(self.root,path[0]+".root"))
            if self.files.has_key(filename):
                file = self.files[filename]
            else:
                file = File(filename)
                self.files[filename] = file
            object = file.Get("/".join(path[1:]))
            if object:
                return (object, filename)
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
                    friendTreeFile = File(friends[0].GetTitle())
                    self.friendFiles[friends[0].GetTitle()] = friendTreeFile
                filename = friends[0].GetTitle()
                tree = friendTreeFile.Get(friendTreeName)
                tree.AddFriend(tmpTree)
            elif len(friends) > 1 and self.verbose:
                print "Warning! This tree has multiple friends!"
        if cuts:
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
    
    def get_samples(self, samplestring, properties = None, **kwargs):
        
        samples = []
        for s in samplestring.split('+'):
            samples.append(self.get_sample(s, properties = properties, **kwargs))
        return samples
    
    def get_sample(self, samplestring, treetype=None, cuts=None, maxEntries=-1, fraction=-1, properties = None):
       
        if self.datasets is None or self.trees is None or self.variables is None:
            return None
        
        samples = [samplestring]
        if "+" in samplestring:
            samples = samplestring.split('+')

        trees = []
        
        for samplestring in samples:
            sample_match = re.match(SAMPLE_REGEX, samplestring)
            if not sample_match:
                raise SyntaxError("%s is not valid sample syntax"% samplestring)
            samplename = sample_match.group('name')
            sampletype = sample_match.group('type')
            if sampletype is None and treetype is None:
                if self.trees.has_key('default'):
                    sampletype = self.trees['default']
                elif len(self.trees) is 1:
                    sampletype = self.trees.values()[0]
                else:
                    raise ValueError("No sample type specified yet no default exists")
            elif (treetype is not None) and (sampletype is not None) and (sampletype != treetype):
                raise ValueError("Conflicting sample types specified: %s and %s"% (sampletype, treetype))
            elif sampletype is None and treetype is not None:
                sampletype = treetype
            if sampletype not in self.trees.keys() and sampletype != 'default':
                raise ValueError("sample type %s is not defined"% sampletype)
            elif sampletype == 'default':
                raise ValueError("sample type cannot be 'default'")
            
            tree_paths, label, datatype, classtype = metadata.find_sample(samplename, sampletype, self.datasets, self.trees)
            trees = []

            for treepath in tree_paths:
                if self.verbose: print "==========================================================="
                trees.append(self.get_tree(treepath, maxEntries=maxEntries, fraction=fraction, cuts=cuts))
            if not trees:
                raise RuntimeError("sample %s (type %s) was not found"% (samplename, treetype))
            for tree in trees:
                if tree is None:
                    raise RuntimeError("sample %s (type %s) was not found"% (samplename, treetype))
                # set aliases
                for branch in self.trees[sampletype]:
                    if tree.GetBranch(branch):
                        if self.variables.has_key(branch):
                            if self.variables[branch].has_key('alias'):
                                tree.SetAlias(self.variables[branch]['alias'],branch)
        
        return Treeset(name = samplename,
                       title = label,
                       label = None,
                       datatype = datatype,
                       classtype = classtype,
                       trees = trees,
                       weight = 1.,
                       tags = None,
                       meta = self.variables,
                       properties = properties)
