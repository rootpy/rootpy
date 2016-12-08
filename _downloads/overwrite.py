#!/usr/bin/env python
"""
======================================
Copy a tree while overwriting branches
======================================

This is an example showing how to copy a tree while overwriting one or more of
its branches with new values.
"""
print(__doc__)
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
from random import gauss

"""
This first section of code only creates an example tree.
"""

# define the model
class Event(TreeModel):
    x = FloatCol()
    y = FloatCol()
    z = FloatCol()
    i = IntCol()

# first create a tree "test" in a file "test.root"
f = root_open("test.root", "recreate")

tree = Tree("test", model=Event)

# fill the tree
for i in range(100):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)
    tree.i = i
    tree.fill()
tree.write()

"""
This section below takes the example tree and copies it while overwriting a
branch with new values.
"""

# Now we want to copy the tree above into a new file while overwriting a branch
# First create a new file to save the new tree in:
f_copy = root_open("test_copy.root", "recreate")

# You may not know the entire model of the original tree but only the branches
# you intend to overwrite, so I am not specifying the model=Event below as an
# example of how to deal with this in general:
tree_copy = Tree("test_copy")

# Here we specify the buffer for the new tree to use. We use the same buffer as
# the original tree. This creates all the same branches in the new tree but
# their addresses point to the same memory used by the original tree.
tree_copy.set_buffer(tree._buffer, create_branches=True)

# Now loop over the original tree and fill the new tree
for entry in tree:
    # Overwrite a branch value. This changes the value that will be written to
    # the new tree but leaves the value unchanged in the original tree on disk.
    entry.x = 3.141
    # "entry" is actually the buffer, which is shared between both trees.
    tree_copy.Fill()

# tree_copy is now a copy of tree where the "x" branch has been overwritten
# with new values
tree_copy.Write()
f_copy.Close()
f.Close()
