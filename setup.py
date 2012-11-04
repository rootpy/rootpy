#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from glob import glob
import os
from os.path import join
import sys

local_path = os.path.dirname(os.path.abspath(__file__))
# setup.py can be called from outside the rootpy directory
os.chdir(local_path)
sys.path.insert(0, local_path)

# check for custom args
# we should instead extend distutils...
filtered_args = []
release = False
build_extensions = True
for arg in sys.argv:
    if arg == '--release':
        # --release sets the version number before installing
        release = True
    elif arg == '--no-ext':
        build_extensions = False
    else:
        filtered_args.append(arg)
sys.argv = filtered_args

ext_modules = []

if os.getenv('ROOTPY_NO_EXT') not in ('1', 'true') and build_extensions:
    from distutils.core import Extension
    import subprocess
    import distutils.sysconfig

    python_lib = os.path.dirname(
                    distutils.sysconfig.get_python_lib(
                        standard_lib=True))

    if 'CPPFLAGS' in os.environ:
        del os.environ['CPPFLAGS']
    if 'LDFLAGS' in os.environ:
        del os.environ['LDFLAGS']

    try:
        root_inc = subprocess.Popen(
            ['root-config', '--incdir'],
            stdout=subprocess.PIPE).communicate()[0].strip()
        root_ldflags = subprocess.Popen(
            ['root-config', '--libs', '--ldflags'],
            stdout=subprocess.PIPE).communicate()[0].strip().split()
        root_cflags = subprocess.Popen(
            ['root-config', '--cflags'],
            stdout=subprocess.PIPE).communicate()[0].strip().split()
    except OSError:
        print('root-config not found. '
              'Please activate your ROOT installation before '
              'the root_numpy extension can be compiled '
              'or set ROOTPY_NO_EXT=1 .')
        sys.exit(1)

    try:
        import numpy as np

        module = Extension(
                'rootpy.root2array.root_numpy._librootnumpy',
                sources=['rootpy/root2array/root_numpy/_librootnumpy.cxx'],
                include_dirs=[np.get_include(),
                              root_inc,
                              'rootpy/root2array/root_numpy/'],
                extra_compile_args=root_cflags,
                extra_link_args=root_ldflags + ['-L%s' % python_lib])
        ext_modules.append(module)

        module = Extension(
                'rootpy.plotting._libnumpyhist',
                sources=['rootpy/plotting/src/_libnumpyhist.cxx'],
                include_dirs=[np.get_include(),
                              root_inc,
                              'rootpy/plotting/src'],
                extra_compile_args=root_cflags,
                extra_link_args=root_ldflags + ['-L%s' % python_lib])
        ext_modules.append(module)

    except ImportError:
        # could not import numpy, so don't build numpy ext_modules
        pass

    module = Extension(
            'rootpy.interactive._pydispatcher_processed_event',
            sources=['rootpy/interactive/src/_pydispatcher.cxx'],
            include_dirs=[root_inc],
            extra_compile_args=root_cflags,
            extra_link_args=root_ldflags + ['-L%s' % python_lib])
    ext_modules.append(module)

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
print __doc__

setup(
    name='rootpy',
    version=__version__,
    description="A pythonic layer on top of the "
    "ROOT framework's PyROOT bindings.",
    long_description=open('README.rst').read(),
    author='Noel Dawe',
    author_email='noel.dawe@cern.ch',
    license='GPLv3',
    url=__url__,
    download_url=__download_url__,
    packages=find_packages(),
    install_requires=[
        'python>=2.6',
        'argparse>=1.2.1',
        ],
    extras_require={
        'hdf': ['tables>=2.3'],
        'array': ['numpy>=1.6.1'],
        'mpl': ['matplotlib>=1.0.1'],
        'term': ['readline>=6.2.4',
                 'termcolor>=1.1.0'],
        },
    scripts=glob('scripts/*'),
    package_data={'': ['etc/*']},
    ext_modules=ext_modules,
    classifiers=[
      "Programming Language :: Python",
      "Topic :: Utilities",
      "Operating System :: POSIX :: Linux",
      "Development Status :: 4 - Beta",
      "Intended Audience :: Science/Research",
      "Intended Audience :: Developers",
      "License :: OSI Approved :: GNU General Public License (GPL)"
    ])

if release:
    # revert rootpy/info.py
    shutil.move('info.tmp', 'rootpy/info.py')
