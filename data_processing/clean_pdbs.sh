#!/bin/bash

# test input
if [ -z "$1" ]
then
  echo "usage: clean_pdbs.sh <pdb_dir>"
  exit
fi

ROSETTADIR=/data/liv/rosetta-3.5

# change dir
PDB_DIR=$1
OLD_DIR=`pwd`
cd $PDB_DIR

# clean PDBs
rm -f *.clean*pdb
PDB_FILES=*.pdb
for f in $PDB_FILES
do
  echo "Cleaning $f"
  $ROSETTADIR/rosetta_tools/protein_tools/scripts/clean_pdb.py $f ignorechain
done

# rename files to *.clean.pdb
rm -f *.fasta
for f in *ignorechain.pdb; do mv $f ${f/_ignorechain/.clean}; done

# change dir back
cd $OLD_DIR
