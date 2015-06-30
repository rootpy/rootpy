#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Check if we are running Python 2 or 3. This is needed for the apt-get package names
if [[ $TRAVIS_PYTHON_VERSION == '3.4' ]]; then
    export PYTHON_SUFFIX="3"
fi

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get -qq update
sudo apt-get -qq install g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 90
sudo apt-get install -qq python${PYTHON_SUFFIX}-pip python${PYTHON_SUFFIX}-numpy python${PYTHON_SUFFIX}-sphinx python${PYTHON_SUFFIX}-nose

# matplotlib and PyTables are not available for Python 3 as packages from the main repo yet.
if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then
    time sudo apt-get install -qq python${PYTHON_SUFFIX}-matplotlib python${PYTHON_SUFFIX}-tables
fi

pip install uncertainties

# Install the ROOT binary
ROOT_BUILD=ROOT-${ROOT}_Python-${TRAVIS_PYTHON_VERSION}_GCC-4.8_x86_64
time wget --no-check-certificate https://copy.com/rtIyUdxgjt7h/ci/root_builds/${ROOT_BUILD}.tar.gz
time tar zxf ${ROOT_BUILD}.tar.gz
mv ${ROOT_BUILD} root
source root/bin/thisroot.sh

# Install the master branch of root_numpy
git clone https://github.com/rootpy/root_numpy.git && (cd root_numpy && python setup.py install)
