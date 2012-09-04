import ROOT
from rootpy.io import open as ropen
from rootpy.tree import Tree

f = ropen("test.root", "recreate")

t = Tree()

f2 = ropen("test2.root", "recreate")
f2.Close()

print ROOT.gDirectory.GetName()

t.Write()
f.Close()
