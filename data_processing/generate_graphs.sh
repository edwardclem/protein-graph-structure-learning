#!/bin/bash

pdbfile=../data/raw_pdb/1aa2.pdb
outfile=../data/graph_files/1aa2_graph.txt

python graph_from_pdb.py -pdb $pdbfile -o $outfile
