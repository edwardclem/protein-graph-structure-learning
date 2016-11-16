#downloading PDB files from ID

from urllib2 import urlopen
from argparse import ArgumentParser
from sys import argv

def parse(args):
	parser = ArgumentParser()
	parser.add_argument("-ids", help="file containing list of PDB ids")
	parser.add_argument("-f", help="download FASTA file", action="store_true")
	parser.add_argument("-p", help="download PDB file", action="store_true")
	parser.add_argument("-o", help="output directory")
	parser.add_argument("-n", help="number to download")
	return parser.parse_args(args)



def run(args):
	with open(args.ids, 'r') as pdblist:
		pdb_ids = [line.split()[0] for line in pdblist]

	if not (args.p or args1):
		print "pick a file type to download!"
	else:
		for pdb_id in pdb_ids[:args.n]:
			if args.f:
				download_pdb(pdb_id, output_directory)
			if args.p:
				download_fasta(pdb_id, output_directory)


def download_fasta(pdb_id, output_directory):
	fasta_url = "https://files.rcsb.org/download/{}.fasta".format(pdb_id)
	f = urlopen(fasta_url)
	print "downloading {}".format(fasta_url)
	with open(output_directory + "/{}.fasta".format(pdb_id), 'wb') as outfile:
		outfile.write(f.read())

def download_pdb(pdb_id, output_directory):
	pdb_url = "https://files.rcsb.org/download/{}.pdb".format(pdb_id)
	f = urlopen(pdb_url)
	print "downloading {}".format(pdb_url)
	with open(output_directory + "/{}.pdb".format(pdb_id), 'wb') as outfile:
		outfile.write(f.read())


if __name__=="__main__":
	run(parse(argv[1:]))



