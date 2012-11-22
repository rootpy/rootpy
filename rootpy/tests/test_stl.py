from rootpy.stl import CPPType
from nose.tools import assert_raises, assert_equal

from rootpy.extern.pyparsing import ParseException

GOOD = [
    'pair<vector<int>, double>',
    'pair<vector<int>, vector<double> >',
    'vector<vector<vector<double> > >',
    'map<int, string>',
    'map<int, vector<double> >',
    'map<int, vector<vector<double> > >',
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

if __name__ == "__main__":
    import nose
    nose.runmodule()
