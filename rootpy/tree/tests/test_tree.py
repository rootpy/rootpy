import ROOT
from rootpy.math.physics.vector import LorentzVector
from rootpy.tree import Tree, TreeModel, TreeChain
from rootpy.io import open as ropen, TemporaryFile
from rootpy.types import FloatCol, IntCol
from random import gauss, randint, random
from nose.tools import assert_raises, assert_almost_equal, with_setup
from unittest import TestCase


class TreeTests(TestCase):

    temp_file = None
    temp_file_path = None

    @classmethod
    def setup_class(cls):

        cls.temp_file = TemporaryFile()
        cls.temp_file_path = cls.temp_file.GetName()

        class ObjectA(TreeModel):
            # A simple tree object
            x = FloatCol()
            y = FloatCol()
            z = FloatCol()

        class ObjectB(TreeModel):
            # A tree object collection
            x = ROOT.vector('int')
            y = ROOT.vector('float')
            vect = ROOT.vector('TLorentzVector')
            # collection size
            n = IntCol()

        class Event(ObjectA.prefix('a_') + ObjectB.prefix('b_')):
            i = IntCol()

        tree = Tree("tree", model=Event)

        # fill the tree
        for i in xrange(10000):
            tree.a_x = gauss(.5, 1.)
            tree.a_y = gauss(.3, 2.)
            tree.a_z = gauss(13., 42.)
            tree.b_vect.clear()
            tree.b_x.clear()
            tree.b_y.clear()
            tree.b_n = randint(1, 5)
            for j in xrange(tree.b_n):
                vect = LorentzVector(
                        gauss(.5, 1.),
                        gauss(.5, 1.),
                        gauss(.5, 1.),
                        gauss(.5, 1.))
                tree.b_vect.push_back(vect)
                tree.b_x.push_back(randint(1, 10))
                tree.b_y.push_back(gauss(.3, 2.))
            tree.i = i
            tree.fill()
        tree.write()
        cls.temp_file.write()

    @classmethod
    def teardown_class(cls):

        cls.temp_file.close()

    def test_attrs(self):

        with ropen(self.temp_file_path) as f:
            tree = f.tree
            tree.use_cache()
            tree.define_object('a', 'a_')
            tree.define_collection('b', 'b_', 'b_n')
            for event in tree:
                # test a setattr before a getattr with caching
                new_a_y = random()
                event.a_y = new_a_y
                assert_almost_equal(event.a_y, new_a_y)

                assert event.a_x == event.a.x
                assert len(event.b) > 0


if __name__ == "__main__":
    import nose
    nose.runmodule()
