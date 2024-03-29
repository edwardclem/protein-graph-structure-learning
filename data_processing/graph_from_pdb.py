#generate graph representations for all PDB files in a folder
#PDB file contains atom positions, amino acids are considered "adjacent" 
#if their carbon alphas are within 10 angstroms from each other
#produces text file with edge list

from argparse import ArgumentParser
from sys import argv
from itertools import combinations, combinations_with_replacement
import numpy as np
import os

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', \
    				'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', \
    				'SER', 'THR', 'TRP', 'TYR', 'VAL']
AMINO_ACIDS = {el : i for i, el in enumerate(AA_LIST)}
AA_IDX = {el : i for i, el in enumerate(combinations_with_replacement(AA_LIST, 2))}

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("-pdb", help="PDB file containing protein structure")
    parser.add_argument("-o", help="Output folder")
    return parser.parse_args(args)

def graph_from_pdb(filename, outfolder):
    #storing ordered tuples containing amino acid type and coordinates
    sequence = []
    amino_number = 0

    #open PDB file
    with open(filename, 'r') as pdbfile:
        for line in pdbfile:
            line = line.split()
            if line[0] == "ATOM" and line[2] == "CA":
                amino_type = line[3]
                coords = tuple(line[6:9]) 
                #TODO: handle ranges of coordinates
                sequence.append((str(amino_number), amino_type, np.array([float(coord) for coord in coords])))
                amino_number = amino_number + 1

    #produce all possible pairs of amino acids, compute distances
    edges = []
    edgecount = 0

    for pair in combinations(sequence, 2):
        #unpack tuples
        amino1, type1, coord1 = pair[0]
        amino2, type2, coord2 = pair[1]

        dist = np.linalg.norm(coord1 - coord2)
        if dist < 10.0: #angstroms
            edges.append((amino1, type1, amino2, type2))

    #convert to string, save file
    base = os.path.basename(filename)
    outfilename = "{}/{}_graph.txt".format(outfolder, os.path.splitext(base)[0])

    #computing length of sequence
    outstr = "{}\n{}".format(len(sequence), "\n".join([" ".join(edge) for edge in edges]))
    print "saving to {}".format(outfilename)
    with open(outfilename, 'w') as outfile:
        outfile.write(outstr)

def features_from_pdb(filename, outfolder):
    #storing ordered tuples containing amino acid type and coordinates
    sequence = []
    amino_number = 0

    #open PDB file
    with open(filename, 'r') as pdbfile:
        for line in pdbfile:
            line = line.split()
            if line[0] == "ATOM" and line[2] == "CA":
                amino_type = line[3]
                coords = tuple(line[6:9]) 
                #TODO: handle ranges of coordinates
                sequence.append((str(amino_number), amino_type, np.array([float(coord) for coord in coords])))
                amino_number = amino_number + 1

    #produce all possible pairs of amino acids, compute distances
    n_aa_combinations = len(AMINO_ACIDS)*(len(AMINO_ACIDS) + 1)/2
    seq_dist_idx = n_aa_combinations
    seq_len_idx = n_aa_combinations + 2
    features = np.zeros((len(sequence)*(len(sequence) + 1)/2, \
    						n_aa_combinations + 1 + 1 + 1)) # 1 for distance, 1 for seqlength, 1 for true example
    features[:,seq_len_idx] = len(sequence)
    for i, pair in enumerate(combinations(sequence, 2)):
        #unpack tuples
        seq_num1, type1, coord1 = pair[0]
        seq_num2, type2, coord2 = pair[1]

        aa_idx = AA_IDX[(min(type1, type2), max(type1, type2))]

        # idx*(idx + 1)/2 is the offset for dealing with upper triangular matrix
        features[i, aa_idx] = 1
        features[i, seq_dist_idx] = abs(int(seq_num1) - int(seq_num2))

        dist = np.linalg.norm(coord1 - coord2)
        if dist < 10.0: #angstroms
        	features[i, -1] = 1
            

    #convert to string, save file
    base = os.path.basename(filename)
    outfilename = "{}/{}_features.npy".format(outfolder, os.path.splitext(base)[0])
    np.save(outfilename, features)

def run(args):
    features_from_pdb(args.pdb, args.o)

if __name__ == "__main__":
    run(parse(argv[1:]))