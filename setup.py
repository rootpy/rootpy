#!/usr/bin/env python
# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

import sys

# check Python version
if sys.version_info < (2, 6):
    sys.exit("rootpy only supports python 2.6 and above")

# check that ROOT can be imported
try:
    import ROOT
except ImportError:
    sys.exit("ROOT cannot be imported. Is ROOT installed with PyROOT enabled?")

ROOT.PyConfig.IgnoreCommandLineOptions = True

# check that we have at least the minimum required version of ROOT
if ROOT.gROOT.GetVersionInt() < 52800:
    sys.exit("rootpy requires at least ROOT 5.28/00; "
             "You have ROOT {0}.".format(ROOT.gROOT.GetVersion()))

import os
# Prevent distutils from trying to create hard links
# which are not allowed on AFS between directories.
# This is a hack to force copying.
try:
    del os.link
except AttributeError:
    pass

try:
    import setuptools
    from pkg_resources import parse_version, get_distribution
    # check that we have setuptools after the merge with distribute
    setuptools_dist = get_distribution('setuptools')
    if setuptools_dist.parsed_version < parse_version('0.7'):
        raise ImportError(
            "setuptools {0} is currently installed".format(
                setuptools_dist.version))
except ImportError as ex:
    sys.exit(
        "{0}\n\n"
        "rootpy requires that at least setuptools 0.7 is installed:\n\n"
        "wget https://bootstrap.pypa.io/ez_setup.py\n"
        "python ez_setup.py --user\n\n"
        "You might need to add the --insecure option to the last command above "
        "if using an old version of wget.\n\n"
        "If you previously had distribute installed, "
        "you might need to manually uninstall the distribute-patched "
        "setuptools before upgrading your setuptools. "
        "See https://pypi.python.org/pypi/setuptools "
        "for further details.".format(ex))

from setuptools import setup, find_packages
from glob import glob
from os.path import join, abspath, dirname, isfile, isdir

local_path = dirname(abspath(__file__))
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
    # remove dev from version in rootpy/info.py
    import shutil
    shutil.move('rootpy/info.py', 'info.tmp')
    dev_info = ''.join(open('info.tmp', 'r').readlines())
    open('rootpy/info.py', 'w').write(
        dev_info.replace('.dev0', ''))

exec(open('rootpy/info.py').read())
if 'install' in sys.argv:
    print(__doc__)

scripts = glob('scripts/*')
if __version__ == 'dev' and devscripts:
    scripts.extend(glob('devscripts/*'))


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        join(os.getcwd(), 'requirements', *f)).readlines()]))


setup(
    name='rootpy',
    version=__version__,
    description="A pythonic layer on top of the "
                "ROOT framework's PyROOT bindings.",
    long_description=''.join(open('README.rst').readlines()[7:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    maintainer='Noel Dawe',
    maintainer_email='noel@dawe.me',
    license='GPLv3',
    url=__url__,
    download_url=__download_url__,
    packages=find_packages(),
    extras_require={
        'tables': reqs('tables.txt'),
        'array': reqs('array.txt'),
        'matplotlib': reqs('matplotlib.txt'),
        'roosh': reqs('roosh.txt'),
        'stats': reqs('stats.txt'),
        },
    scripts=scripts,
    entry_points={
        'console_scripts': [
            'root2hdf5 = rootpy.root2hdf5:main',
            'roosh = rootpy.roosh:main',
            ]
        },
    package_data={'': [
        'etc/*',
        'testdata/*.root',
        'testdata/*.txt',
        'tests/test_compiled.cxx',
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
