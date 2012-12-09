from rootpy.plotting.style import Style, get_style
from ROOT import TStyle


def test_get_style():

    mystyle = TStyle('mystyle', 'some style')
    assert(isinstance(get_style('mystyle'), Style))

if __name__ == "__main__":
    import nose
    nose.runmodule()
