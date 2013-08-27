# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.utils.silence import silence
from nose.tools import assert_equal


def test_silence_with_exception():

    try:
        with silence() as filt_content:
            print "Hmm"
            raise RuntimeError("Error!")
    except:
        pass
    assert_equal(filt_content.getvalue(), 'Hmm\n')


if __name__ == "__main__":
    import nose
    nose.runmodule()
