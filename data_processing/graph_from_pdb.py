#generate a graph representation from a PDB file
#PDB file contains atom positions, amino acids are considered "adjacent" 
#if their carbon alphas are within 10 angstroms from each other
#produces text file with edge list

from argparse import ArgumentParser
from sys import argv
from itertools import combinations
import numpy as np


def parse(args):
	parser = ArgumentParser()
	parser.add_argument("-pdb", help="PDB file containing protein structure")
	parser.add_argument("-o", help="Output file")
	return parser.parse_args(args)

def run(args):

	#storing ordered tuples containing amino acid type and coordinates
	sequence = []
	amino_number = 0

	#open PDB file
	with open(args.pdb, 'r') as pdbfile:
		for line in pdbfile:
			line = line.split()
			if line[0] == "ATOM" and line[2] == "CA":
				amino_type = line[3]
				coords = tuple(line[6:9]) 
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

	outstr = "\n".join([" ".join(edge) for edge in edges])

	with open(args.o, 'w') as outfile:
		outfile.write(outstr)

if __name__ == "__main__":
	run(parse(argv[1:]))