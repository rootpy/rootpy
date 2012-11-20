from rootpy.stl import parse_template
from nose.tools import assert_raises, assert_equal


GOOD = [
    'pair<vector<int>,double>',
    'pair<vector<int>,vector<double> >',
    'vector<vector<vector<double> > >',
    'map<int,string>',
    'map<int,vector<double> >',
    'map<int,vector<vector<double> > >',
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
        assert_equal(template, str(parse_template(template)))
    for template in BAD:
        assert_raises(SyntaxError, parse_template, template)

if __name__ == "__main__":
    import nose
    nose.runmodule()
