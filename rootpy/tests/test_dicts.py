import ROOT
from pkg_resources import resource_filename
from rootpy.io import open as ropen
from rootpy.rootcint import generate
from rootpy import stl
from rootpy.stl import CPPType
from nose.plugins.attrib import attr


@attr('slow')
def test_rootcint():

    generate('map<int,vector<float> >', '<vector>;<map>')
    generate('map<int,vector<int> >', '<vector>;<map>')
    generate('vector<TLorentzVector>', '<vector>;TLorentzVector.h')

    ROOT.std.map('int,vector<float>')
    ROOT.std.map('int,vector<int>')
    ROOT.std.vector('TLorentzVector')


    temp = CPPType.from_string('vector<vector<vector<int> > >')
    temp.ensure_built()

    stl.vector('vector<map<int, string> >')

    stl.map("string", "string")

    stl.map(stl.string, stl.string)
    stl.map(int, stl.string)
    stl.map(stl.string, int)

    stl.map("string", ROOT.TLorentzVector)

    histmap = stl.map("string", ROOT.TH1D)()
    a = ROOT.TH1D("a", "a", 10, -1, 1)
    histmap["a"] = a

    StrHist = stl.pair(stl.string, "TH1*")

    histptrmap = stl.map(stl.string, "TH1*")()
    histptrmap.insert(StrHist("test", a))

    assert histptrmap["test"] is a


def test_dict_load():

    filename = resource_filename('rootpy', 'etc/test_dicts.root')
    with ropen(filename) as f:
        # this will trigger the loading of the dicts required by all branches
        f.data


if __name__ == "__main__":
    import nose
    nose.runmodule()
