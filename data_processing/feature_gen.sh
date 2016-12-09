#!/bin/bash
#$ -S /bin/sh
#$ -cwd
#$ -r y
#$ -o ../log/$JOB_ID.out
#$ -e ../log/$JOB_ID.err

#runs feature generation on one file
python feat_from_pdb.py -pdb $1 -o $2 -d $3
