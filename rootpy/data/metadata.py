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
        # the string is the yml?
        return yaml.load(string)

def get_variable_meta(name, meta):

    for var,details in meta.items():
        if details.has_key('alias'):
            if name == details['alias']:
                return details
        if name == var:
            return details
    return None

def find_sample(samplename, treetype, datasets, objects, label=None, classtype=None, datatype=None, tree_paths = None):
    """
    Recursively find the first dataset with a name matching samplename
    """
    if tree_paths == None:
        tree_paths = []
    if type(datasets) is not dict:
        return tree_paths, label, classtype, datatype
    for key, value in datasets.items():
        if type(value) is dict:
            if value.has_key('class'):
                classtype = value['class']
            if value.has_key('type'):
                datatype = value['type']
            if value.has_key('label'):
                label = value['label']
        if key == samplename:
            _recurse_find_trees(treetype, value, key, objects, tree_paths)
            return tree_paths, label, classtype, datatype
        else:
            find_sample(samplename, treetype, value, objects, label, classtype, datatype, tree_paths)
    return tree_paths, label, classtype, datatype

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
