#!/bin/bash

DROPBOX=~/Dropbox/Public/builds

if [ ! -e $DROPBOX ]; then
    echo "first mount dropbox at ${DROPBOX}"
    exit 1
fi

PYTHON=2.7
BASE=${HOME}/software/root/ # leave the trailing / here

for ROOT in v5-34-08 v5-32-04
do
    if [ -e $DROPBOX/root_${ROOT}_python_${PYTHON}.tar.gz ]; then
        echo "the build $DROPBOX/root_${ROOT}_python_${PYTHON}.tar.gz already exists"
        continue
    fi

    if [ ! -e $BASE ]; then
        mkdir -p $BASE
    fi
    cd $BASE
    if [ ! -e root ]; then
        git clone http://root.cern.ch/git/root.git || exit 1
    fi

    if [ ! -e root_${ROOT}_python_${PYTHON} ]; then
        cd root
        make clean
        git branch -D $ROOT
        git checkout -b $ROOT $ROOT
        ./configure --enable-python --enable-roofit --enable-tmva --disable-xrootd --fail-on-missing || exit 1
        make -j 4 || exit 1
        export ROOTSYS=root_${ROOT}_python_${PYTHON}
        make DESTDIR=$BASE install || exit 1
        cd ..
    fi

    tar zcvf root_${ROOT}_python_${PYTHON}.tar.gz root_${ROOT}_python_${PYTHON}
    mv root_${ROOT}_python_${PYTHON}.tar.gz $DROPBOX
done
