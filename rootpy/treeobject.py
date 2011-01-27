from collections import namedtuple

def treeobject(name, tree, prefix):

    class name: pass

class TreeObjectCollection(list):

    def __init__(self, name, tree, prefix):

        self.name = name
        self.tree = tree
