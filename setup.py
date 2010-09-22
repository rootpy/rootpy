#!/usr/bin/env python

# Place current directory at the front of PYTHONPATH
import sys
sys.path.insert(0,'.')

from ROOTPy import pkginfo
from distutils.core import setup
from distutils.command.install_data import install_data
from glob import glob
import os
import re

# generates files using templates and install them
class install_data_pyroot (install_data):
    def initialize_options (self):
        install_data.initialize_options (self)
        self.prefix = None
        self.root   = None
        self.install_purelib = None
        self.install_scripts = None

    def finalize_options (self):
        # set install_purelib
        self.set_undefined_options('install',
                                   ('prefix','prefix'))
        self.set_undefined_options('install',
                                   ('root','root'))
        self.set_undefined_options('install',
                                   ('install_purelib','install_purelib'))
        self.set_undefined_options('install',
                                   ('install_scripts','install_scripts'))
                                            
    def run (self):
        rpmInstall = False
        # set install_dir
        if self.install_dir == None:
            if self.root != None:
                # rpm
                self.install_dir = self.root
                rpmInstall = True
            else:
                # sdist
                self.install_dir = self.prefix
        self.install_dir = os.path.expanduser(self.install_dir)
        self.install_dir = os.path.abspath(self.install_dir)
        # remove /usr for bdist/bdist_rpm
        match = re.search('(build/[^/]+/dumb)/usr',self.install_dir)
        if match != None:
            self.install_dir = re.sub(match.group(0),match.group(1),self.install_dir)
        # remove /var/tmp/*-buildroot for bdist_rpm
        match = re.search('(/var/tmp/.*-buildroot)/usr',self.install_dir)
        if match != None:
            self.install_dir = re.sub(match.group(0),match.group(1),self.install_dir)
        # create tmp area
        tmpDir = 'build/tmp'
        self.mkpath(tmpDir)
        new_data_files = []
        for destDir,dataFiles in self.data_files:
            newFilesList = []
            for srcFile in dataFiles:
                print srcFile
                # dest filename
                destFile = re.sub('\.template$','',srcFile)
                destFile = destFile.split('/')[-1]
                destFile = '%s/%s' % (tmpDir,destFile)
                # open src
                inFile = open(srcFile)
                # read
                filedata=inFile.read()
                # close
                inFile.close()
                # replace patterns
                for item in re.findall('@@([^@]+)@@',filedata):
                    if not hasattr(self,item):
                        raise RuntimeError,'unknown pattern %s in %s' % (item,srcFile)
                    # get pattern
                    patt = getattr(self,item)
                    # convert to absolute path
                    if item.startswith('install'):
                        patt = os.path.abspath(patt)
                    # remove build/*/dump for bdist
                    patt = re.sub('build/[^/]+/dumb','',patt)
                    # remove /var/tmp/*-buildroot for bdist_rpm
                    patt = re.sub('/var/tmp/.*-buildroot','',patt)
                    # replace
                    filedata = filedata.replace('@@%s@@' % item, patt)
                # write to dest
                oFile = open(destFile,'w')
                oFile.write(filedata)
                oFile.close()
                # append
                newFilesList.append(destFile)
            # replace dataFiles to install generated file
            new_data_files.append((destDir,newFilesList))
        # install
        self.data_files = new_data_files
        install_data.run(self)

setup(name='ROOTPy',
      version=pkginfo.release,
      description='ROOT utilities',
      author='Noel Dawe',
      author_email='noel.dawe@cern.ch',
      url='http://noel.mine.nu/repo',
      packages=['ROOTPy', 'ROOTPy.analysis'],
      requires=['ROOT','multiprocessing'],
      scripts=glob('scripts/*'),
      data_files = [('etc',glob('templates/*'))],
      cmdclass={'install_data':install_data_pyroot},
     )

