#!/bin/bash

#downloads FASTA files for each PDB id

pdb_file=../data/author.idx
output_directory=../data/fasta

python download_pdb -ids $pdb_file -f -o $output_directory