#!/usr/bin/env python

import sys, os
import numpy as np
from itertools import combinations

def potential_test(filename):
	seqlen = 0
	edges = set()
	with open(filename) as f:
		seqlen = int(f.readline())
		line = f.readline()
		while line:
			line = line.split(' ')
			first = int(line[0])
			second = int(line[2])
			edges.add((first, second))
			line = f.readline()
	#print len(edges)
	edge_density = np.zeros(4)
	for (first, second, third) in combinations(range(seqlen), 3):
		num_present = int((first, second) in edges)
		num_present += int((first, third) in edges)
		num_present += int((second, third) in edges)
		edge_density[num_present] += 1
	return edge_density, len(edges), seqlen*(seqlen - 1)/2

def get_three_factor_stats(edges, seqlen, cutoff):
	edge_density = np.zeros(4, dtype=int)
	for (first, second) in combinations(range(seqlen), 2):
		for third in range(seqlen):
			if (first == third) or (second == third):
				continue
			#if all three are observed, ignore
			if not((first - third) < cutoff and (first - second) < cutoff and (first - third) < cutoff):
				num_present = int((first, second) in edges)
				num_present += int((first, third) in edges)
				num_present += int((second, third) in edges)
				edge_density[num_present] += 1
	return edge_density

def test_on_files(file_list, outfile=None):
	print "test_on_files"
	edge_density = np.zeros((len(file_list), 4))
	total_edges = np.zeros(len(file_list))
	possible_edges = np.zeros(len(file_list))
	for i, filename in enumerate(file_list):
		print "{:0.1f} percent complete".format(100*float(i)/len(file_list))
		file_edge_dense, n_edges, n_possible = potential_test(filename)
		edge_density[i,:] = file_edge_dense
		total_edges[i] = n_edges
		possible_edges[i] = n_possible
	print "100 percent complete"
	if outfile:
		np.save(outfile, np.concatenate([edge_density, total_edges, possible_edges]))
	else:
		return edge_density, total_edges, possible_edges

def analyze_edge_dense(edge_density, n_edges, n_possible):
	appearance_prob = n_edges / n_possible
	expected_frac = [(1 - appearance_prob)**3, 3*(1 - appearance_prob)**2*appearance_prob, \
						3*(1 - appearance_prob)*appearance_prob**2, appearance_prob**3]
	expected_frac = np.array(expected_frac).T
	true_frac = edge_density / np.sum(edge_density, 1)[:,None]
	prob_ratio = true_frac / expected_frac
	empirical_prob = np.sum(edge_density, 0)/np.sum(edge_density)
	return empirical_prob, prob_ratio

	

if __name__ == '__main__':
	print "main"
	directory = sys.argv[1]
	file_list = os.listdir(directory)
	test_on_files([directory + fname for fname in file_list], sys.argv[2])
	
