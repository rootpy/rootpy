#!/bin/bash

sudo apt-get install git
sudo apt-get install dpkg-dev make g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev libxml2-dev
sudo apt-get install python-dev python-nose python-numpy python-matplotlib python-tables ipython-notebook
sudo apt-get install python3-dev python3-nose python3-numpy ipython3
# to install older pythons
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python2.6 python2.6-dev
# also see https://github.com/utahta/pythonbrew
