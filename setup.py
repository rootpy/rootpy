#!/usr/bin/env python

# Place current directory at the front of PYTHONPATH
import sys
sys.path.insert(0,'.')

from distutils.core import setup
from glob import glob

setup(name='PyROOT',
      version='1.0',
      description='ROOT utilities',
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url='http://noel.mine.nu/repo',
      packages=['PyROOT', 'PyROOT.analysis'],
      requires=['ROOT','multiprocessing'],
      scripts=glob("scripts/*"),
     )

