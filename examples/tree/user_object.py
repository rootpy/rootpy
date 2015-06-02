#!/usr/bin/env python
"""
================================================
Create trees with branches of user-defined types
================================================

This example demonstrates how to fill and read trees with branches containing
user-defined types.
"""
print(__doc__)
from rootpy.tree import Tree, TreeModel, IntCol, ObjectCol
from rootpy.io import root_open
import rootpy.compiled as C
from random import gauss

# compile our new type
C.register_code("""

class Thingy {
    public:
        int i;
        float x;
        double y;
};

""", ["Thingy"])

# alternatively you can ROOT.gSystem.Load() your library

# define the model
class Event(TreeModel):
    event_number = IntCol()
    thingy = ObjectCol(C.Thingy)


f = root_open("test.root", "recreate")
tree = Tree("test", model=Event)

# fill the tree
for i in range(20):
    tree.event_number = i
    tree.thingy.i = i
    tree.thingy.x = gauss(.3, 2.)
    tree.thingy.y = gauss(13., 42.)
    tree.fill()

tree.write()
f.close()

# now to read the same tree
with root_open("test.root") as f:
    tree = f.test
    for event in tree:
        thing = event.thingy
        print("{0} {1} {2} {3}".format(
            event.event_number, thing.i, thing.x, thing.y))
