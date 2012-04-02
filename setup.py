#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from glob import glob
import os

ext_modules = []

try:
    import numpy as np
    from distutils.core import Extension
    import distutils.util
    import subprocess

    root_inc = subprocess.Popen(["root-config", "--incdir"],
                                stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen(["root-config", "--libs"],
                                stdout=subprocess.PIPE).communicate()[0].strip().split()

    module = Extension('rootpy.root2array.root_numpy._librootnumpy',
                        sources=['rootpy/root2array/root_numpy/_librootnumpy.cxx'],
                        include_dirs=[np.get_include(), root_inc],
                        #extra_compile_args = root_cflags,
                        extra_link_args=root_ldflags)
    ext_modules.append(module)
except ImportError:
    # could not import numpy, so don't build numpy ext_modules
    pass

execfile('rootpy/info.py')

setup(name='rootpy',
      version=__VERSION__,
      description='The way PyROOT should be, and more!',
      long_description=open('README.rst').read(),
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url=__URL__,
      download_url=__DOWNLOAD_URL__,
      packages=find_packages(),
      install_requires = ['python>=2.6',
                          'argparse'],
      scripts=glob('scripts/*') + \
              glob('rootpy/scripts-standalone/*'),
      package_data={'': ['scripts-standalone/*', 'etc/*']},
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
