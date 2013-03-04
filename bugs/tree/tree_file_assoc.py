import ROOT
from rootpy.io import root_open as ropen
from rootpy.tree import Tree

f = ropen("test.root", "recreate")

print ROOT.gDirectory.GetName()

t = Tree()

f2 = ropen("test2.root", "recreate")
f2.Close()

print ROOT.gDirectory.GetName()

#f.cd() <== this should not be needed!
# the tree should "remember" what file it was created in
t.Write()
f.Close()
