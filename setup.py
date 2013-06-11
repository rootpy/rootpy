#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

import os
# Prevent distutils from trying to create hard links
# which are not allowed on AFS between directories.
# This is a hack to force copying.
try:
    del os.link
except AttributeError:
    pass

try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

from glob import glob
from os.path import join
import sys

if sys.version_info < (2, 6):
    raise RuntimeError("rootpy only supports python 2.6 and above")

local_path = os.path.dirname(os.path.abspath(__file__))
# setup.py can be called from outside the rootpy directory
os.chdir(local_path)
sys.path.insert(0, local_path)

# check for custom args
# we should instead extend distutils...
filtered_args = []
release = False
devscripts = False
for arg in sys.argv:
    if arg == '--release':
        # --release sets the version number before installing
        release = True
    elif arg == '--devscripts':
        devscripts = True
    else:
        filtered_args.append(arg)
sys.argv = filtered_args

if release:
    # write the version to rootpy/info.py
    version = open('version.txt', 'r').read().strip()
    import shutil
    shutil.move('rootpy/info.py', 'info.tmp')
    dev_info = ''.join(open('info.tmp', 'r').readlines())
    open('rootpy/info.py', 'w').write(
            dev_info.replace(
                "version_info('dev')",
                "version_info('%s')" % version))

execfile('rootpy/info.py')
if 'install' in sys.argv:
    print __doc__

scripts = glob('scripts/*')
if __version__ == 'dev' and devscripts:
    scripts.extend(glob('devscripts/*'))

def strip_comments(l):
    return l.split('#', 1)[0].strip()

def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        os.path.join(os.getcwd(), 'requirements', *f)).readlines()]))

def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py')))

def find_packages(path='.', base=''):
    """ Find all packages in path """
    packages = {}
    for item in os.listdir(path):
        dirpath = os.path.join(path, item)
        if is_package(dirpath):
            if base:
                module_name = '%(base)s.%(item)s' % vars()
            else:
                module_name = item
            packages[module_name] = dirpath
            packages.update(find_packages(dirpath, module_name))
    return packages

packages = find_packages()

setup(
    name='rootpy',
    version=__version__,
    description="A pythonic layer on top of the "
    "ROOT framework's PyROOT bindings.",
    long_description=''.join(open('README.rst').readlines()[8:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    license='GPLv3',
    url=__url__,
    download_url=__download_url__,
    package_dir=packages,
    packages=packages.keys(),
    extras_require={
        'tables': reqs('tables.txt'),
        'array': reqs('array.txt'),
        'matplotlib': reqs('matplotlib.txt'),
        'roosh': reqs('roosh.txt'),
        'fit': reqs('fit.txt'),
        },
    scripts=scripts,
    entry_points={
        'console_scripts': [
            'root2hdf5 = rootpy.root2hdf5:main',
            ]
        },
    package_data={'': [
        'etc/*',
        'testdata/*.root',
        'testdata/*.txt',
        'compiled/tests/test.cxx',
        ]},
    classifiers=[
      'Programming Language :: Python',
      'Programming Language :: Python :: 2',
      'Topic :: Utilities',
      'Operating System :: POSIX :: Linux',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License (GPL)'
    ])

if release:
    # revert rootpy/info.py
    shutil.move('info.tmp', 'rootpy/info.py')
