#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from glob import glob
import os
import sys

ext_modules = []

if os.getenv('ROOTPY_NO_EXT') not in ('1', 'true'):
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
                'rootpy.root2array._libnumpyhist',
                sources=['rootpy/root2array/src/_libnumpyhist.cxx'],
                include_dirs=[np.get_include(),
                              root_inc,
                              'rootpy/root2array/src'],
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

execfile('rootpy/info.py')

print __doc__

setup(name='rootpy',
      version=__version__,  # @UndefinedVariable
      description="A pythonic layer on top of the "
      "ROOT framework's PyROOT bindings.",
      long_description=open('README.rst').read(),
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url=__url__,  # @UndefinedVariable
      download_url=__download_url__,  # @UndefinedVariable
      packages=find_packages(),
      install_requires=['python>=2.6',
                        'argparse'],
      scripts=glob('scripts/*'),
      package_data={'': ['etc/*']},
      ext_modules=ext_modules,
      license='GPLv3',
      classifiers=[
        "Programming Language :: Python",
        "Topic :: Utilities",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License (GPL)"
      ]
     )
