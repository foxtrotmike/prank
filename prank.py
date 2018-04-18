# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:43:29 2014
Use this file to generate pRANK predictions
@author: fayyaz
"""

from MI2SVM import *
from lamp_nr import testFasta
from features import *
import glob
import argparse
if __name__=="__main__": 
	parser = argparse.ArgumentParser(description='Use pRANK for predicting FASTA files.')
	parser.add_argument('ifile', type=str, help='Input FASTA File')
	parser.add_argument('--ofile', type=str, help='Output MIP File', default = None)
	args = parser.parse_args()
	classifier=MI2SVM()
	classifier.load('aac_yeast_proteome.cls')
	ifile = args.ifile
	ofile = args.ofile
	if args.ofile is None:
		ofile=ifile+".mip"		
	HWS=20
	testFasta(classifier,pfile=None,nfile=None,HWS=HWS,tfile=ifile,ofile=ofile,comm=None,nprocs=1,myid=0)