#generate graph representations for all PDB files in a folder
#PDB file contains atom positions, amino acids are considered "adjacent" 
#if their carbon alphas are within 10 angstroms from each other
#produces text file with edge list

from argparse import ArgumentParser
from sys import argv
from itertools import combinations, combinations_with_replacement
import numpy as np
import scipy.io as sio
import os
from scipy import sparse, io
from test_three_edge_potential import get_three_factor_stats

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', \
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', \
                    'SER', 'THR', 'TRP', 'TYR', 'VAL']
AMINO_ACIDS = {el : i for i, el in enumerate(AA_LIST)}
AA_IDX = {el : i for i, el in enumerate(combinations_with_replacement(AA_LIST, 2))}

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("-pdb", help="PDB file containing protein structure")
    parser.add_argument("-d", help="lower limit for edge distance", type=int)
    parser.add_argument("-o", help="Output folder")
    return parser.parse_args(args)

def features_from_pdb(filename, outfolder, dist_cutoff):
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
                if len(amino_type) == 4:
                    amino_type = amino_type[1:]
                sequence.append((str(amino_number), amino_type, np.array([float(coord) for coord in coords])))
                amino_number = amino_number + 1

    num_edges = len(sequence)*(len(sequence) - 1)/2

    #produce all possible pairs of amino acids, compute distances
    n_aa_combinations = len(AMINO_ACIDS)*(len(AMINO_ACIDS) + 1)/2
    amino_acid_index = np.zeros((num_edges,1), dtype=int)
    seq_dist_total = 0
    true_example = np.zeros((num_edges, 1), dtype=int)

    edges = set()

    for i, pair in enumerate(combinations(sequence, 2)):
        #unpack tuples
        seq_num1, type1, coord1 = pair[0]
        seq_num2, type2, coord2 = pair[1]

        #ignore edges smaller than the cutoff
        if abs(int(seq_num1) - int(seq_num2)) > dist_cutoff:
            aa_pair = AA_IDX[(min(type1, type2), max(type1, type2))]

            # idx*(idx + 1)/2 is the offset for dealing with upper triangular matrix
            amino_acid_index[i] = aa_pair

            dist = np.linalg.norm(coord1 - coord2)
            if dist < 10.0: #angstroms
                edges.add(tuple(sorted((int(seq_num1), int(seq_num2)))))
                true_example[i] = 1
                seq_dist_total += abs(int(seq_num1) - int(seq_num2))
    edge_density = get_three_factor_stats(edges, len(sequence), dist_cutoff)

    #suff_stats_protein = np.concatenate((features, np.array([num_edges], dtype=int), edge_density))

    #convert to string, save file
    base = os.path.basename(filename)
    outfile = "{}/{}_processed.mat".format(outfolder, os.path.splitext(base)[0])
    print "saving to {}".format(outfile)
    sio.savemat(outfile, {'aa_index': amino_acid_index, 'edge_density':edge_density, 'sum_true_dist':seq_dist_total, 'true_edges' : true_example, 'seqlen':len(sequence)})
    #saving as sparse matrix
    # outfile_ss = "{}/{}_features.mat".format(outfolder, os.path.splitext(base)[0])
    # print "saving to {}".format(outfile_ss)
    # #io.mmwrite(outfilename, sparse.lil_matrix(features))
    # #np.save(outfile_ss, suff_stats_protein)
    # sio.savemat(outfile_ss, {'ss_protein' : suff_stats_protein})
    # outfile_true = "{}/{}_true.mat".format(outfolder, os.path.splitext(base)[0])
    # print "saving truth to {}".format(outfile_true)
    # #np.save(outfile_true, true_example)
    # sio.savemat(outfile_true, {})



def run(args):
    features_from_pdb(args.pdb, args.o, args.d)

if __name__ == "__main__":
    run(parse(argv[1:]))
