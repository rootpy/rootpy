# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import ROOT
from rootpy import stl
from rootpy.stl import CPPType, generate
from rootpy.testdata import get_file
from rootpy.extern.pyparsing import ParseException

from nose.plugins.attrib import attr
from nose.tools import assert_raises, assert_equal

from multiprocessing import Pool


GOOD = [
    'std::pair<vector<const int*>, double>*',
    'pair<vector<int>, vector<double> >',
    'vector<vector<vector<double> > >::iterator*',
    'map<int, string>',
    'map<int, vector<double> >',
    'map<int, vector<vector<double> > >',
    'vector<unsigned int>',
    'vector<const int*>',
    'vector<unsigned int>',
]

BAD = [
    'pair<vector<int>,double>>',
    'pair<vector<int>,,vector<double> >',
    'vector<<vector<vector<double> > >',
    'int,string',
    'int,vector<double> >',
    'vector<double> >',
    'map<int,vector<vector<double> > >,',
]


def test_parse():
    for template in GOOD:
        assert_equal(template, str(CPPType.from_string(template)))
    for template in BAD:
        assert_raises(ParseException, CPPType.from_string, template)


@attr('slow')
def test_stl():
    generate('map<int,vector<float> >', '<vector>;<map>')
    generate('map<int,vector<int> >', '<vector>;<map>')
    generate('vector<TLorentzVector>', '<vector>;TLorentzVector.h')

    ROOT.std.map('int,vector<float>')
    ROOT.std.map('int,vector<int>')
    ROOT.std.vector('TLorentzVector')

    temp = CPPType.from_string('vector<vector<vector<int> > >')
    temp.ensure_built()

    stl.vector('vector<map<int, string> >')
    stl.vector(stl.string)()
    stl.vector('string')()
    stl.vector(int)

    stl.map("string", "string")
    stl.map(stl.string, stl.string)
    stl.map(int, stl.string)
    stl.map(stl.string, int)
    stl.map("string", ROOT.TLorentzVector)

    histmap = stl.map("string", ROOT.TH1D)()
    a = ROOT.TH1D("a", "a", 10, -1, 1)
    histmap["a"] = a

    StrHist = stl.pair(stl.string, "TH1*")

    generate('pair<map<string,TH1*>::iterator,bool>', '<map>;<TH1.h>')
    histptrmap = stl.map(stl.string, "TH1*")()
    histptrmap.insert(StrHist("test", a))

    assert histptrmap["test"] is a

"""
This test frequently fails on Travis due to os.fork() not being able to
allocate memory. Disabling it for now until a solution is found.

def load_tree(*args):
    with get_file('test_dicts.root') as f:
        t = f.data
        # this will trigger the generation of the required dicts
        t.create_buffer()


def test_dict_load():
    # test file locking
    po = Pool()
    po.map(load_tree, range(3))
"""

if __name__ == "__main__":
    import nose
    nose.runmodule()
