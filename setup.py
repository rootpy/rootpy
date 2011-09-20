#!/usr/bin/env python

from rootpy import pkginfo
from distutils.core import setup
from glob import glob

setup(name='rootpy',
      version=pkginfo.__RELEASE__,
      description='The way PyROOT should be, and more!',
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url='http://ndawe.github.com/rootpy',
      packages=['rootpy', 'rootpy.plotting', 'rootpy.data', 'rootpy.tree', 'rootpy.utils', 'rootpy.io', 'rootpy.batch'],
      requires=['ROOT', 'matplotlib', 'numpy', 'PyYAML'],
      scripts=glob('scripts/*')
     )

