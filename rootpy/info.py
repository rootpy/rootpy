# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
"""
                 _
 _ __ ___   ___ | |_ _ __  _   _
| '__/ _ \ / _ \| __| '_ \| | | |
| | | (_) | (_) | |_| |_) | |_| |
|_|  \___/ \___/ \__| .__/ \__, |
                    |_|    |___/
      %s
"""
from __future__ import absolute_import

from collections import namedtuple


_version_info_base = namedtuple(
    'version_info',
    ['major',
     'minor',
     'micro'])


class version_info(_version_info_base):

    DEV = (999, 9, 9)

    def __new__(cls, version):

        if version == 'dev':
            return super(version_info, cls).__new__(cls, *version_info.DEV)
        else:
            return super(version_info, cls).__new__(cls, *version.split('.'))

    def __repr__(self):

        return 'rootpy.%s' % super(version_info, self).__repr__()

    def __str__(self):

        if self == version_info.DEV:
            return 'dev'
        return '%s.%s.%s' % self


__version_info__ = version_info('dev')
__version__ = str(__version_info__)
__url__ = 'http://rootpy.github.com/rootpy'
__repo_url__ = 'https://github.com/rootpy/rootpy/'
__download_url__ = ('http://pypi.python.org/packages/source/r/'
                    'rootpy/rootpy-%s.tar.gz') % __version__
__doc__ %= __version__
