#!/bin/bash

DROPBOX=~/Dropbox/Public/builds

if [ ! -e $DROPBOX ]; then
    echo "first mount dropbox at ${DROPBOX}"
    exit 1
fi

ROOT=v5-34-08
PYTHON=2.7
BASE=${HOME}/software/root/ # leave the trailing / here

if [ -e $DROPBOX/root_${ROOT}_python_${PYTHON}.tar.gz ]; then
    echo "the build $DROPBOX/root_${ROOT}_python_${PYTHON}.tar.gz already exists"
    exit 0
fi

if [ ! -e $BASE ]; then
    mkdir -p $BASE
fi
cd $BASE
if [ ! -e root ]; then
    git clone http://root.cern.ch/git/root.git
fi

if [ ! -e root_${ROOT}_python_${PYTHON} ]; then
    cd root
    make clean
    git checkout -b $ROOT $ROOT
    ./configure --enable-roofit
    make -j 4
    export ROOTSYS=root_${ROOT}_python_${PYTHON}
    make DESTDIR=$BASE install
    cd ..
fi

tar zcvf root_${ROOT}_python_${PYTHON}.tar.gz root_${ROOT}_python_${PYTHON}
mv root_${ROOT}_python_${PYTHON}.tar.gz $DROPBOX
