#!/bin/bash


#produces a graph file from all PDB files in the given folder
pdbfolder=../data/raw_pdb
outfolder=../data/graph_files

for file in $pdbfolder/*; do	
	python graph_from_pdb.py -pdb $file -o $outfolder
done


