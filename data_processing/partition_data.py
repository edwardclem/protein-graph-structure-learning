#partitions data in a given folder into the provided training and test folders
#checking to see if isomers are encountered

from argparse import ArgumentParser
from sys import argv
from os import listdir
from random import random
from os.path import isdir
from shutil import copyfile

def parse(args):
	parser = ArgumentParser()
	parser.add_argument("-f", help="Folder with data")
	parser.add_argument("-t", help="Percentage of files used for training data", type=float)
	parser.add_argument("--train", help="folder for training data")
	parser.add_argument("--test", help="folder for test data")
	return parser.parse_args(args)

def run(args):
	#iterating through all files in folder


	files = [file for file in listdir(args.f) if not isdir(file)]
	isomers = set() #getting unique proteins (i.e. filtering out isomers)
	unique_proteins = []
	for file in files:
		prot_name = file.split("_")[0]
		prot_prefix = prot_name[0:3]
		if not prot_prefix in isomers:
			isomers.add(prot_name[0:3]) #first three letters unique
			unique_proteins.append(file)

	for protein in unique_proteins:
		sourcefile = "{}/{}".format(args.f, protein)
		if (random() < args.t):
			#copy to train
			destfile = "{}/{}".format(args.train, protein)
		else:
			destfile = "{}/{}".format(args.test, protein)
		print "saving {} to {}".format(sourcefile, destfile)
		copyfile(sourcefile, destfile)


if __name__=="__main__":
	run(parse(argv[1:]))