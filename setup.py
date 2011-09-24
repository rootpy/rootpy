#!/usr/bin/env python

from distutils.core import setup
from urlparse import urljoin
from glob import glob

execfile('rootpy/info.py')

setup(name='rootpy',
      version=__VERSION__,
      description='The way PyROOT should be, and more!',
      long_description=open('README.rst').read(),
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url=__URL__,
      download_url=urljoin(__GITURL__, "tarball/master"),
      packages=['rootpy',
                'rootpy.plotting',
                'rootpy.data',
                'rootpy.tree',
                'rootpy.utils',
                'rootpy.io',
                'rootpy.batch',
                'rootpy.hep'],
      requires=['ROOT', 'matplotlib', 'numpy', 'PyYAML'],
      scripts=glob('scripts/*'),
      license='GPLv3',
      classifiers=[
        "Programming Language :: Python",
        "Topic :: Utilities",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License (GPL)"
      ]
     )
