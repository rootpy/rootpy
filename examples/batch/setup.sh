#!/bin/bash

cd files
for i in 1 2 3 4;
do
    cp file.root file${i}.root
done 
cd -
