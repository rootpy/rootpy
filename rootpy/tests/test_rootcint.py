import ROOT
from rootpy.rootcint import generate
from rootpy import stl
from rootpy.stl import parse_template


def test_rootcint():

    generate('map<int,vector<float> >', '<vector>;<map>', verbose=True)
    generate('map<int,vector<int> >', '<vector>;<map>', verbose=True)
    generate('vector<TLorentzVector>', '<vector>;TLorentzVector.h', verbose=True)

    ROOT.std.map('int,vector<float>')
    ROOT.std.map('int,vector<int>')
    ROOT.std.vector('TLorentzVector')


    temp = parse_template('vector<vector<vector<int> > >')
    temp.compile(True)

    stl.vector('vector<map<int, string> >', verbose=True)


if __name__ == "__main__":
    import nose
    nose.runmodule()
