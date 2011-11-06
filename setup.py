#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
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
      packages=find_packages(),
      install_requires = ['python>=2.6', 'argparse'],
      scripts=glob('scripts/*') + \
              glob('rootpy/scripts-standalone/*'),
      package_data={'': ['scripts-standalone/*']},
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
