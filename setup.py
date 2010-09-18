#!/usr/bin/env python

from distutils.core import setup
from glob import glob

setup(name='PyROOT',
      version='1.0',
      description='ROOT utilities',
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url='http://noel.mine.nu/repo',
      package_dir = {'PyROOT':'lib'},
      packages=['PyROOT', 'PyROOT.analysis'],
      requires=['ROOT','multiprocessing'],
      scripts=glob("scripts/*"),
     )

