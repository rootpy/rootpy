# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from rootpy.testdata import get_file

from nose.tools import assert_equal, with_setup
from nose.plugins.skip import SkipTest

from tempfile import mkdtemp
import os
import shutil


TEMPDIR = None


def setup_func():
    global TEMPDIR
    TEMPDIR = mkdtemp()


def teardown_func():
    shutil.rmtree(TEMPDIR)


@with_setup(setup_func, teardown_func)
def test_root2hdf5():
    try:
        import tables
    except ImportError:
        raise SkipTest

    from rootpy.root2hdf5 import root2hdf5

    rfile = get_file('test_tree.root')
    hfilename = os.path.join(TEMPDIR, 'out.h5')
    root2hdf5(rfile, hfilename)

    hfile = tables.openFile(hfilename)
    assert_equal(len(hfile.root.test), 1000)
    hfile.close()


@with_setup(setup_func, teardown_func)
def test_root2hdf5_chunked():
    try:
        import tables
    except ImportError:
        raise SkipTest

    from rootpy.root2hdf5 import root2hdf5

    rfile = get_file('test_tree.root')
    hfilename = os.path.join(TEMPDIR, 'out.h5')
    root2hdf5(rfile, hfilename, entries=10)

    hfile = tables.openFile(hfilename)
    assert_equal(len(hfile.root.test), 1000)
    hfile.close()


@with_setup(setup_func, teardown_func)
def test_root2hdf5_chunked_selected():
    try:
        import tables
    except ImportError:
        raise SkipTest

    from rootpy.root2hdf5 import root2hdf5

    rfile = get_file('test_tree.root')
    hfilename = os.path.join(TEMPDIR, 'out.h5')
    root2hdf5(rfile, hfilename, entries=90, selection='i % 2 == 0')

    hfile = tables.openFile(hfilename)
    assert_equal(len(hfile.root.test), 500)
    hfile.close()


if __name__ == "__main__":
    import nose
    nose.runmodule()
