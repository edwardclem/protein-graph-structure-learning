#!/bin/bash

#produces a graph file from all PDB files in the given folder
pdbfolder=../../data/raw_pdb
outfolder=../../data/summed_suffstats

#parallel --jobs 2 --eta python feat_from_pdb.py -pdb {1} -o $outfolder ::: $pdbfolder/*.pdb

#using for loop instead
for file in $pdbfolder/*; do	
	python feat_from_pdb.py -pdb $file -o $outfolder
done