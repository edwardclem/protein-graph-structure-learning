#plotting a protein interaction graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#returns a dictionary of amino acid type for each edge
def get_edge_labels(edgelist):
	labels = {}

	for edge in edgelist:
		if edge[0] not in labels.keys():
			labels[edge[0]] = edge[1]
		if edge[2] not in labels.keys():
			labels[edge[2]] = edge[3]

	return labels

def run():
	edgefile = "../data/graph_files/1aa2_graph.txt"

	with open(edgefile, 'r') as edgefile:
		edgelist = [edge.split() for edge in edgefile]

		edge_labels = get_edge_labels(edgelist)

		edges_processed = [(edge[0], edge[2]) for edge in edgelist]


		G = nx.Graph()
		G.add_edges_from(edges_processed)

		pos = nx.spring_layout(G, k=3/np.sqrt(len(edge_labels.keys())))
		nx.draw_networkx(G, pos=pos)

		plt.show()





if __name__ =="__main__":
	run()