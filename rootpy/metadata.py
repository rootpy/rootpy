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

def find_sample(samplename, treetype, datasets, objects, classtype=None, datatype=None, tree_paths = None):
    """
    Recursively find the first dataset with a name matching samplename
    """
    if tree_paths == None:
        tree_paths = []
    if type(datasets) is not dict:
        return tree_paths, classtype, datatype
    for key, value in datasets.items():
        if type(value) is dict:
            if value.has_key('class'):
                classtype = value['class']
            if value.has_key('type'):
                datatype = value['type']
        if key == samplename:
            _recurse_find_trees(treetype, value, key, objects, tree_paths)
            return tree_paths, classtype, datatype
        else:
            find_sample(samplename, treetype, value, objects, classtype, datatype, tree_paths)
    return tree_paths, classtype, datatype

def _recurse_find_trees(treetype, datasets, parent, objects, tree_paths):
    """
    Recursively find all trees of type treetype within the scope of this dataset
    """
    for key, value in datasets.items():
        if not type(value) is dict:
            continue
        if value.has_key('tree'):
            if value['tree'] == treetype:
                tree_paths.append("%s/%s"% (parent, key))
        else:
            _recurse_find_trees(treetype, value, key, objects, tree_paths)
