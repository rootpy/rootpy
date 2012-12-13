# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import os
from pkg_resources import resource_filename
from ..io import File


def get_file(name='test_file.root'):
    filename = resource_filename('rootpy', os.path.join('testdata', name))
    if not os.path.isfile(filename):
        raise ValueError('rootpy data file %s does not exist' % filename)
    return File(filename, 'read')
