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
