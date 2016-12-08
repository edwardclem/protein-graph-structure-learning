#!/bin/bash

datfolder=../data/summed_suffstats
trainfolder=../data/train
testfolder=../data/test

python partition_data.py -f $datfolder -t 0.75 --train $trainfolder --test $testfolder