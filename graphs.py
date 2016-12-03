import os

from random import randint
import numpy as np

"""
Some random graph related class/method things. 

Copied them over from other projects as they may be useful.
"""
class Vertex:
	def __init__(self, aa):
		self.edges = set()
		self.aa = None

	def __str__(self):
		return 'AA: ' + str(self.aa) + ', Edges: ' + str(self.edges)

def printgraph(graph):
	keys = sorted(list(graph))
	for key in keys:
		print graph[key]

def reprgraph(graph):
	keys = sorted(list(graph))
	if np.isscalar(keys[0]) or len(keys[0]) == 1:
		arr = np.zeros(len(keys))
		for key in keys:
			arr[key] = graph[key].aa
	elif len(keys[0]) == 2:
		arr = np.zeros((np.sqrt(len(keys)), np.sqrt(len(keys))))
		for key in keys:
			arr[key[::-1]] = graph[key].aa
	else:
		raise ValueError("Can only handle 1d or 2d graphs")
	return arr

def make_graph(filename):
	graph = {}
	with open(filename) as f:
		for line in f:
			line = line.split(' ')
			idx = int(line[0])
			aa = line[1]
			nbr = int(line[2])
			nbr_aa = line[3]
			if idx in graph:
				graph[idx].edges.add(nbr)
			else:
				vertex = Vertex(aa)
				vertex.edges.add(nbr)
				graph[idx] = vertex
			if nbr in graph:
				graph[nbr].edges.add(idx)
			else:
				vertex = Vertex(nbr_aa)
				vertex.edges.add(idx)
				graph[nbr] = vertex
	return graph

# Makes count of number of edges that appear at each distance
# Works if seq_length is not provided at start of graph file
def get_distance_count_deprecated(filename):
	observed = {}
	seq_length = 0
	with open(filename) as f:
		for line in f:
			line = line.split(' ')
			curr = int(line[0])
			seq_length = curr
			other = int(line[2])
			dist = abs(curr - other)
			observed[dist] = observed.get(dist, 0) + 1
	seq_length = seq_length + 2
	observed = np.array([observed.get(i, 0) for i in xrange(1, seq_length)], dtype=np.float64)
	total = np.arange(seq_length, dtype=np.float64)[:0:-1]
	return observed, total

# Makes count of number of edges that appear at each distance
# Must put seq_length at beginning of graph file
def get_distance_count(filename):
	observed = None
	with open(filename) as f:
		seq_length = int(f.readline())
		observed = np.zeros(seq_length - 1, dtype=np.float64)
		line = f.readline()
		while line:
			line = line.split(' ')
			curr = int(line[0])
			other = int(line[2])
			dist = abs(curr - other)
			observed[dist - 1] += 1
			line = f.readline()
	total = np.arange(len(observed) + 1, dtype=np.float64)[:0:-1]
	return observed, total

# Gets count over all graph files in a directory. Cannot have anything else in directory.
def get_distance_all(directory, num_files=None):
	total_observed = np.zeros(1)
	relative_distance = []
	total_count = np.zeros(1)
	for i, filename in enumerate(os.listdir(directory)):
		observed, count = get_distance_count(directory + filename)
		relative_distance.append(observed/len(observed))
		if (len(total_observed) >= len(observed)):
			total_observed[:len(observed)] += observed
			total_count[:len(observed)] += observed
		else:
			observed[:len(total_observed)] += total_observed
			count[:len(total_observed)] += total_count
			total_observed = observed
			total_count = count
		if num_files and i >= num_files:
			break
	relative_distance = np.concatenate(relative_distance)
	return total_observed, total_count, relative_distance

import matplotlib.pyplot as plt
def show_graph(protein_name):
	filename = 'data/graph_files/' + protein_name + '_graph.txt'
	observed, count = get_distance_count(filename)
	plt.plot(observed/count)
	plt.show()




