#!/bin/bash

# rootpy account at www.copy.com
CLOUD=~/Copy/ci/root_builds

if [ ! -e $CLOUD ]; then
    echo "first mount dropbox at ${CLOUD}"
    exit 1
fi

PYTHON=2.7
BASE=${HOME}/software/root/ # leave the trailing / here
if [ ! -e $BASE ]; then
    mkdir -p $BASE
fi
cd $BASE
if [ ! -e root ]; then
    git clone http://root.cern.ch/git/root.git || exit 1
fi

if [ ! -e tags.lst ]; then
    echo "list the tags, one per line in tags.lst"
fi

while read ROOT;
do
    if [ -e ${CLOUD}/root_${ROOT}_python_${PYTHON}.tar.gz ]; then
        echo "${CLOUD}/root_${ROOT}_python_${PYTHON}.tar.gz already exists"
        continue
    fi
    
    echo "building $ROOT ..."

    cd root
    make clean
    git checkout master
    git branch -D $ROOT
    git checkout -b $ROOT $ROOT
    ./configure --enable-python --enable-roofit --enable-xml --enable-tmva --disable-xrootd --fail-on-missing || exit 1
    make -j 4 || exit 1
    export ROOTSYS=root_${ROOT}_python_${PYTHON}
    make DESTDIR=$BASE install || exit 1
    cd ..

    tar zcvf root_${ROOT}_python_${PYTHON}.tar.gz root_${ROOT}_python_${PYTHON}
    mv root_${ROOT}_python_${PYTHON}.tar.gz $CLOUD

done < tags.lst
