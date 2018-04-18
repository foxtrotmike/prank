# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:19:06 2014

@author: fayyaz
"""
import add2path

import numpy as np
import random, pdb
try:
    import PyML
    from PyML.classifiers.baseClassifiers import Classifier
    from mpi4py import MPI
except ImportError:
    print "PyML/MPI Import Failed"
    
import matplotlib.pyplot as plt


import os,sys



from papa import *
from myPickle import *
import re
import myPickle

#load sequences for prions



from features import *
from evaluation import *
from data import *        
from MI2SVM import *
#from MI1SVM import *
#from wrapMISVM import *
#from mySVM import *
from ensemble import *
    

def testFasta(classifierTemplate,pfile,nfile,HWS,tfile,ofile=None,comm=None,nprocs=1,myid=0):
    if myid==0:
        if type(tfile)==type(''):
            NR=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(tfile, "rU"), "fasta") if len(record.seq.tostring())>2*HWS+1]        
        else:
            NR = tfile
        if pfile is not None:
            S,A=loadAnnotatedPrions(ifile=pfile)
            PS=featureExtraction(S,HWS=HWS)
            #F=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-HWS))],ps[:,a[1]-HWS:]))) for ps,a in zip(PS,A)]
            F=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-2*HWS))],ps[:,a[1]+HWS:]))) for ps,a in zip(PS,A)]
        if nfile is not None:
            NS=[record.seq.tostring() for record in SeqIO.parse(open(nfile, "rU"), "fasta")]    
            Xn=featureExtraction(NS,HWS=HWS)
        if pfile is not None and nfile is not None:
            classifier=classifierTemplate.__class__(classifierTemplate)
            classifier.train(F,Xn)              
            print "Classifier training done."
        else:
            print "Using trained classifier."
            classifier=classifierTemplate
        for i in range(1,nprocs):
            comm.send((classifier,NR[(i*len(NR))/nprocs:((i+1)*len(NR))/nprocs]),dest=i)
        mine=NR[(myid*len(NR))/nprocs:((myid+1)*len(NR))/nprocs]        
    else:
        classifier,mine=comm.recv(source=0)
    print "Myid is",myid,"$",len(mine)
    S=testOnSeqs(classifier,HWS,mine)
    if myid==0:
        for i in range(1,nprocs):
            S.extend(comm.recv(source=i))
        
        if ofile is not None:     
            writePAPAFile(ofile,S)
            print "Output file:",ofile,"Written."
        return classifier,S
    else:
        comm.send(S , dest=0)
        
if __name__=="__main__":    
    HWS=20#int(sys.argv[1])
    try:
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
        nprocs = comm.Get_size()  
    except NameError:
        comm = None
        myid = 0
        nprocs = 1
    nfile='../data/nonprions.fasta'#'../data/sabate_neg.fasta'#'../data/nonprions_18_complete.fasta'#
    pfile='../data/prions_yeast_22ns.fasta'#'../data/sabate.fasta'#'../data/prion_proteins_annotated_yhf.fasta'#'../data/prions_yeast_22.fasta'#'../data/prion_proteins_annotated_18.fasta'
    classifierTemplate=MI2SVM()
    classifierTemplate.Lambda=5e-6
    classifierTemplate.Epochs=500
    classifierTemplate=classifierEnsemble(classifierTemplate,N=2)
#    RX=testFasta(classifierTemplate,pfile,nfile,HWS,tfile='../data/proteome_yeast.fasta',ofile='proteome_yeast.txt',comm=comm,nprocs=nprocs,myid=myid)
#    if myid==0:
#        classifier,S=RX
#        WX=np.mean([c.W/(c.max-c.min) for c in classifier.classifiers],axis=0)
#        W=WX[:20]
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        rects1 = ax.bar(np.arange(len(W))   , W, 0.8  )
#        ax.set_ylabel('Weight values')
#        plt.xticks(np.arange(len(AA))+0.5,[a for a in AA])
#        plt.grid()
#        plt.show()

#    print S
    print "LOOCV started for",classifierTemplate
    
    RX=loocvpar(classifierTemplate,HWS=HWS,pfile=pfile,nfile=nfile,ofile='loocv40_complete2.txt',comm=comm,nprocs=nprocs,myid=myid)
#    if myid==0:
#        (FH,ps,ns,xx)=RX