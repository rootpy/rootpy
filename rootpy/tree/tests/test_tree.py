# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT

from rootpy.vector import LorentzVector
from rootpy.tree import Tree, Ntuple, TreeModel, TreeChain
from rootpy.io import root_open, TemporaryFile
from rootpy.tree.treetypes import FloatCol, IntCol
from rootpy.plotting import Hist, Hist2D, Hist3D
from rootpy import testdata
from rootpy import stl

from random import gauss, randint, random
import re
import os
import sys
if sys.version_info[0] < 3:
    from cStringIO import StringIO
else:
    from io import StringIO

from nose.plugins.skip import SkipTest
from nose.tools import (assert_raises, assert_almost_equal,
                        assert_equal, raises, with_setup)


FILES = []
FILE_PATHS = []


def create_model():

    class ObjectA(TreeModel):
        # A simple tree object
        x = FloatCol()
        y = FloatCol()
        z = FloatCol()
        vect = LorentzVector

    class ObjectB(TreeModel):
        # A tree object collection
        x = stl.vector('int')
        y = stl.vector('float')
        vect = stl.vector('TLorentzVector')
        # collection size
        n = IntCol()

    class Event(ObjectA.prefix('a_') + ObjectB.prefix('b_')):
        i = IntCol()

    return Event


def create_tree():
    f = TemporaryFile()
    tree = Tree("tree", model=create_model())
    # fill the tree
    for i in range(1000):
        assert_equal(tree.a_vect, LorentzVector(0, 0, 0, 0))
        random_vect = LorentzVector(
            gauss(.5, 1.),
            gauss(.5, 1.),
            gauss(.5, 1.),
            gauss(.5, 1.))
        tree.a_vect.copy_from(random_vect)
        assert_equal(tree.a_vect, random_vect)
        tree.a_x = gauss(.5, 1.)
        tree.a_y = gauss(.3, 2.)
        tree.a_z = gauss(13., 42.)
        tree.b_n = randint(1, 5)
        for j in range(tree.b_n):
            vect = LorentzVector(
                gauss(.5, 1.),
                gauss(.5, 1.),
                gauss(.5, 1.),
                gauss(.5, 1.))
            tree.b_vect.push_back(vect)
            tree.b_x.push_back(randint(1, 10))
            tree.b_y.push_back(gauss(.3, 2.))
        tree.i = i
        assert_equal(tree.b_n, tree.b_vect.size())
        assert_equal(tree.b_n, tree.b_x.size())
        assert_equal(tree.b_n, tree.b_y.size())
        tree.fill(reset=True)
    tree.write()
    # TFile.Close the file but keep the underlying
    # tempfile file descriptor open
    ROOT.TFile.Close(f)
    FILES.append(f)
    FILE_PATHS.append(f.GetName())


def create_chain():
    for i in range(3):
        create_tree()


def cleanup():
    global FILES
    global FILE_PATHS

    for f in FILES:
        f.close()

    FILES = []
    FILE_PATHS = []


@with_setup(create_tree, cleanup)
def test_attrs():
    with root_open(FILE_PATHS[0]) as f:
        tree = f.tree
        tree.read_branches_on_demand = True
        tree.define_object('a', 'a_')
        tree.define_collection('b', 'b_', 'b_n')
        for event in tree:
            # test a setattr before a getattr with caching
            new_a_y = random()
            event.a_y = new_a_y
            assert_almost_equal(event.a_y, new_a_y)

            assert_equal(event.a_x, event.a.x)
            assert_equal(len(event.b) > 0, True)


@with_setup(create_tree, cleanup)
def test_draw():
    with root_open(FILE_PATHS[0]) as f:
        tree = f.tree

        tree.draw('a_x')
        tree.draw('a_x:a_y')
        tree.draw('a_x:TMath::Exp(a_y)')
        tree.draw('a_x:a_y:a_z')
        tree.draw('a_x:a_y:a_z:b_x')
        tree.draw('a_x:a_y:a_z:b_x:b_y', options='para')

        h1 = Hist(10, -1, 2, name='h1')
        h2 = Hist2D(10, -1, 2, 10, -1, 2)
        h3 = Hist3D(10, -1, 2, 10, -1, 2, 10, -1, 2)

        # dimensionality does not match
        assert_raises(TypeError, tree.draw, 'a_x:a_y', hist=h1)

        # name does not match
        assert_raises(ValueError, tree.draw, 'a_x>>+something', hist=h1)

        # hist is not a TH1
        assert_raises(TypeError, tree.draw, 'a_x:a_y', hist=ROOT.TGraph())

        # name does match and is fine (just redundant)
        tree.draw('a_x>>h1', hist=h1)
        assert_equal(h1.Integral() > 0, True)
        h1.Reset()
        tree.draw('a_x>>+h1', hist=h1)
        assert_equal(h1.Integral() > 0, True)
        h1.Reset()

        # both binning and hist are specified
        assert_raises(ValueError, tree.draw, 'a_x>>+h1(10, 0, 1)', hist=h1)

        tree.draw('a_x', hist=h1)
        assert_equal(h1.Integral() > 0, True)
        tree.draw('a_x:a_y', hist=h2)
        assert_equal(h2.Integral() > 0, True)
        tree.draw('a_x:a_y:a_z', hist=h3)
        assert_equal(h3.Integral() > 0, True)

        h3.Reset()
        tree.draw('a_x>0:a_y/2:a_z*2', hist=h3)
        assert_equal(h3.Integral() > 0, True)

        # create a histogram
        hist = tree.draw('a_x:a_y:a_z', create_hist=True)
        assert_equal(hist.Integral() > 0, True)

        hist = tree.draw('a_x:a_y:a_z>>new_hist_1')
        assert_equal(hist.Integral() > 0, True)
        assert_equal(hist.name, 'new_hist_1')

        # create_hist=True is redundant here
        hist = tree.draw('a_x:a_y:a_z>>new_hist_2', create_hist=True)
        assert_equal(hist.Integral() > 0, True)
        assert_equal(hist.name, 'new_hist_2')

        # test list/tuple expression
        hist1 = tree.draw('a_x:a_y:a_z', create_hist=True)
        hist2 = tree.draw(['a_x', 'a_y', 'a_z'], create_hist=True)
        hist3 = tree.draw(('a_x', 'a_y', 'a_z'), create_hist=True)
        assert_equal(hist1.Integral() > 0, True)
        assert_equal(hist2.Integral(), hist1.Integral())
        assert_equal(hist3.Integral(), hist1.Integral())


@with_setup(create_chain, cleanup)
def test_chain_draw():
    if sys.version_info[0] >= 3:
        raise SkipTest("Python 3 support not implemented")
    chain = TreeChain('tree', FILE_PATHS)
    hist = Hist(100, 0, 1)
    chain.draw('a_x', hist=hist)
    assert_equal(hist.Integral() > 0, True)

    # check that Draw can be repeated
    hist2 = Hist(100, 0, 1)
    chain.draw('a_x', hist=hist2)
    assert_equal(hist.Integral(), hist2.Integral())


@with_setup(create_chain, cleanup)
def test_chain_draw_hist_init_first():
    if sys.version_info[0] >= 3:
        raise SkipTest("Python 3 support not implemented")
    hist = Hist(100, 0, 1)
    chain = TreeChain('tree', FILE_PATHS)
    chain.draw('a_x', hist=hist)
    assert_equal(hist.Integral() > 0, True)


@raises(RuntimeError)
def test_require_file_bad():
    t = Tree()


def test_require_file_good():
    with TemporaryFile():
        t = Tree()


@raises(RuntimeError)
def test_require_file_not_writable():
    with testdata.get_file():
        t = Tree()


def test_draw_regex():
    p = Tree.DRAW_PATTERN
    m = re.match
    assert_equal(m(p, 'a') is not None, True)
    assert_equal(m(p, 'somebranch') is not None, True)
    assert_equal(m(p, 'x:y') is not None, True)
    assert_equal(m(p, 'xbranch:y') is not None, True)
    assert_equal(m(p, 'x:y:z') is not None, True)

    expr = '(x%2)>0:sqrt(y)>4:z/3'
    assert_equal(m(p, expr) is not None, True)

    redirect = '>>+histname(10, 0, 1)'
    expr_redirect = expr + redirect
    match = m(p, expr_redirect)
    groupdict = match.groupdict()
    assert_equal(groupdict['branches'], expr)
    assert_equal(groupdict['redirect'], redirect)
    assert_equal(groupdict['name'], 'histname')


def test_file_assoc():
    with TemporaryFile() as f1:
        t = Tree()
        with TemporaryFile() as f2:
            pass
        #f1.cd() <== this should not be needed!
        # the tree should "remember" what file it was created in
        t.Write()


def test_csv():
    f = testdata.get_file('test_csv.root')
    tree = f.ParTree_Postselect
    tree.create_buffer(ignore_unsupported=True)
    output = StringIO()
    tree.csv(stream=output)
    f.close()
    # compare with existing txt output
    if sys.version_info[0] < 3:
        true_output_filename = testdata.get_filepath('test_csv.txt')
    else:
        true_output_filename = testdata.get_filepath('test_csv_new.txt')
    with open(true_output_filename, 'r') as true_output_file:
        true_output = true_output_file.read()
        assert_equal(output.getvalue(), true_output)


def test_ntuple():
    with TemporaryFile():
        ntuple = Ntuple(('a', 'b', 'c'), name='test')
        for i in range(100):
            ntuple.Fill(gauss(.3, 2.), gauss(0, 1.), gauss(-1., 5))
        ntuple.Write()


if __name__ == '__main__':
    import nose
    nose.runmodule()
