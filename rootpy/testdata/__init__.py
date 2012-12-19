# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import os
from pkg_resources import resource_filename
from ..io import File


def get_filepath(name='test_file.root'):
    return resource_filename('rootpy', os.path.join('testdata', name))


def get_file(name='test_file.root'):
    filepath = get_filepath(name)
    if not os.path.isfile(filepath):
        raise ValueError('rootpy data file %s does not exist' % filepath)
    return File(filepath, 'read')
