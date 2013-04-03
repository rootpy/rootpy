#!/usr/bin/env python

from rootpy.tree import Tree, TreeModel
from rootpy.io import root_open
from rootpy.types import FloatCol, ObjectCol
from rootpy.math.physics.vector import LorentzVector
from random import gauss, randint
import ROOT

f = root_open("test.root", "recreate")


# define the model
class Event(TreeModel):

    x = FloatCol()
    y = FloatCol()
    z = FloatCol()

    a = ROOT.vector("float")
    b = ROOT.vector("float")
    c = ROOT.vector("float")

    d = ROOT.vector("vector<float>")
    e = ROOT.vector("vector<float>")
    f = ROOT.vector("vector<float>")

    g = ObjectCol(LorentzVector)

tree = Tree("test", model=Event)

# fill the tree
for i in xrange(50000):
    tree.x = gauss(.5, 1.)
    tree.y = gauss(.3, 2.)
    tree.z = gauss(13., 42.)

    for i in xrange(randint(1, 10)):
        tree.a.push_back(gauss(.5, 1.))
    for i in xrange(randint(1, 10)):
        tree.b.push_back(gauss(.5, 1.))
    for i in xrange(randint(1, 10)):
        tree.c.push_back(gauss(.5, 1.))

    for i in xrange(randint(1, 10)):
        t = ROOT.vector("float")()
        for j in xrange(randint(1, 10)):
            t.push_back(gauss(.5, 1.))
        tree.d.push_back(t)
    for i in xrange(randint(1, 10)):
        t = ROOT.vector("float")()
        for j in xrange(randint(1, 10)):
            t.push_back(gauss(.5, 1.))
        tree.e.push_back(t)
    for i in xrange(randint(1, 10)):
        t = ROOT.vector("float")()
        for j in xrange(randint(1, 10)):
            t.push_back(gauss(.5, 1.))
        tree.f.push_back(t)

    tree.g.SetPtEtaPhiM(2, 2, 2, 2)
    tree.fill(reset=True)
tree.write()

f.close()
