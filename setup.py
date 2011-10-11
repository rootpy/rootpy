#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup
from glob import glob

execfile('rootpy/info.py')

setup(name='rootpy',
      version=__VERSION__,
      description='The way PyROOT should be, and more!',
      long_description=open('README.rst').read(),
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url=__URL__,
      download_url=__DOWNLOAD_URL__,
      packages=['rootpy',
                'rootpy.plotting',
                'rootpy.data',
                'rootpy.tree',
                'rootpy.io',
                'rootpy.batch',
                'rootpy.hep',
                'rootpy.types',
                'rootpy.backports.argparse'],
      requires=['ROOT', 'matplotlib', 'numpy', 'PyYAML'],
      scripts=glob('scripts/*'),
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
