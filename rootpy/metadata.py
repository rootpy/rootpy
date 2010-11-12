"""
Load metadata about datasets from yml (YAML)
"""
import yaml

def load(string):
    """
    Retrieve the variable information
    """
    try:
        # attempt to read as filename
        f = open(string,'r')
        m =  yaml.load(f)
        f.close()
        return m
    except IOError:
        # the string is the xml?
        return yaml.load(string)

def find_sample(samplename, sampletype, datasets, objects):
    """
    Retrieve the tree names and paths, and sample type and class
    """
    tree_paths = []
    classtype, datatype = _recurse_find_sample(samplename, sampletype, datasets, objects, tree_paths)
    return tree_paths, classtype, datatype

def _recurse_find_sample(samplename, treetype, datasets, objects, tree_paths, classtype=None, datatype=None):
    """
    Recursively find the first dataset with a name matching samplename
    """
    if not type(datasets) is dict:
        return classtype, datatype
    for key, value in datasets.items():
        if classtype == None and key == "class":
            classtype = value
        elif datatype == None and key == "type":
            datatype = value
        elif key == samplename:
            _recurse_find_trees(treetype, value, key, objects, tree_paths)
            return classtype, datatype
        else:
            _recurse_find_sample(samplename, sampletype, datasets, objects, tree_paths, classtype, datatype)
    return classtype, datatype

def _recurse_find_trees(treetype, datasets, parent, objects, tree_paths):
    """
    Recursively find all trees of type treetype within the scope of this dataset
    """
    for key, value in datasets.items():
        if value.has_key('tree'):
            if value['tree'] == treetype:
                tree_paths.append("%s/%s"% (parent, key))
        else:
            _recurse_find_trees(treetype, datasets, parent, objects, tree_paths):
