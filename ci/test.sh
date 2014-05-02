#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

# Check if ROOT and PyROOT work
root -l -q
python -c "import ROOT; ROOT.TBrowser()"

# Check that rootpy can be imported
time python -c 'import rootpy'
# What if ROOT has already been initialized?
time python -c 'from ROOT import kTRUE; import rootpy'

# Give user write access to shared memory to make multiprocessing semaphares work 
# https://github.com/rootpy/rootpy/pull/176#issuecomment-13712313
ls -la /dev/shm
sudo rm -rf /dev/shm && sudo ln -s /run/shm /dev/shm
#- sudo chmod a+w /dev/shm
ls -la /dev/shm

# Now run the actual tests (from the installed version, not the local build dir)
time make install-user
time make test-installed
