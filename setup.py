#!/usr/bin/env python

from rootpy import pkginfo
from distutils.core import setup
from glob import glob

setup(name='rootpy',
      version=pkginfo.__RELEASE__,
      description='ROOT utilities',
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url='http://noeldawe.github.com/rootpy',
      packages=['rootpy', 'rootpy.plotting', 'rootpy.data', 'rootpy.tree', 'rootpy.utils', 'rootpy.io', 'rootpy.batch'],
      requires=['ROOT', 'multiprocessing', 'yaml', 'matplotlib', 'numpy'],
      scripts=glob('scripts/*')
     )

