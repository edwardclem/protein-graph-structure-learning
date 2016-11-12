#downloading PDB files from ID

from urllib2 import urlopen

def run(pdb_list, output_directory, num_download):
	with open(pdb_file, 'r') as pdblist:
		pdb_ids = [line.split()[0] for line in pdblist]

	for pdb_id in pdb_ids[:num_download]:
		download_pdb(pdb_id, output_directory)


def download_pdb(pdb_id, output_directory):

	pdb_url = "https://files.rcsb.org/download/{}.pdb".format(pdb_id)
	f = urlopen(pdb_url)
	print "downloading {}".format(pdb_url)
	with open(output_directory + "/{}.pdb".format(pdb_id), 'wb') as outfile:
		outfile.write(f.read())


if __name__=="__main__":
	pdb_file = "../data/author.idx"
	output_directory = "../data/raw_pdb"
	run(pdb_file, output_directory, 1000)

