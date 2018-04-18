# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:07:38 2014
Functions for performance evaluation
@author: fayyaz
"""
import random
import numpy as np
try:
    import PyML
    from PyML.evaluators import roc
except ImportError:
    import roc
from Bio import SeqIO
from data import *
from features import *
import pdb
import tempfile,os
def getFileParts(fname):
    "Returns the parts of a file"
    (path, name) = os.path.split(fname)
    n=os.path.splitext(name)[0]
    ext=os.path.splitext(name)[1]
    return (path,n,ext)
def testOnSeqs(classifier,HWS,mine):
    """
    Give
    """    
    return [(hdr,classifier.testBag(featureExtraction([seq],HWS=HWS)[0],pos=True,vec=True)) for hdr,seq in mine]

def loocvpar(classifierTemplate,HWS,pfile,nfile,ofile=None,comm=None,nprocs=1,myid=0):
    """
    Parallel execution of leave one out cross validation
    Note: Return only from processor id 0 is valid
    Arguments:
        classifierTemplate: classifier Template
        F: Positive bags (list of tuples of numpy arrays)
        N: Negative bags (list of numpy arrays)
        (comm,nprocs,myid): mpi4py arguments
    Return: 
        (FH,ps,ns):
            FH: False hit percentage of all positive examples
            ps: scores of positives
            ns: scores of negatives
    """
    F=None
    N=None
    NRp=None
    NRn=None
    if myid==0:        
        NRp=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(pfile, "rU"), "fasta")]        
        NRn=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(nfile, "rU"), "fasta")]    
        S,A=loadAnnotatedPrions(ifile=pfile)
        PS=featureExtraction(S,HWS=HWS)
        NS=[record.seq.tostring() for record in SeqIO.parse(open(nfile, "rU"), "fasta")]    
        N=featureExtraction(NS,HWS=HWS)
        
        #F=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-HWS))],ps[:,a[1]-HWS:]))) for ps,a in zip(PS,A)]
        F=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-2*HWS))],ps[:,a[1]+HWS:]))) for ps,a in zip(PS,A)]
#        from writeBags import unwriteBagFile, writeBagFile
#        
#        writeBagFile(F,N,ofname = 'prions40_pd1_21.lem')
#        import pdb; pdb.set_trace()    
#
#        F1,N1 = unwriteBagFile('prions40_pd1_21.lem')
#        print np.max(abs(N1[0].toarray()-N[0])),np.max(abs(N1[-1].toarray()-N[-1]))
#        print np.max(abs(F1[0][0].toarray()-F[0][0])),np.max(abs(F1[0][1].toarray()-F[0][1]))
#        print np.max(abs(F1[-1][0].toarray()-F[-1][0])),np.max(abs(F1[-1][1].toarray()-F[-1][1]))
        #from spectrum import *
        #sps = PDSpectrumizer(k = 1, hws = 20)
#    import pdb; pdb.set_trace()    

    if comm is not None:
        F,N,NRp,NRn=comm.bcast((F,N,NRp,NRn),root=0)
    myF=range((myid*len(F))/nprocs,((myid+1)*len(F))/nprocs)
    print "Myid is",myid,"$",len(F),":",myF
    my_ps=[]
    my_FH=[]
    my_xx=[]
    for k in myF:        
        pidx = NRp[k][0].split()[0].upper()
        Ftr=F[:k]+F[k+1:]
        Ftt=F[k]
        classifier = classifierTemplate.__class__(classifierTemplate)
        
        classifier.train(Ftr,N)
        out_file = tempfile.NamedTemporaryFile(prefix='yeast_pos_'+pidx+'_',suffix='.cls');
        ofname=getFileParts(out_file.name)[1]+'.cls'; out_file.close();        
        classifier.save(ofname)
        print "Wrote file: ",ofname
        nsi=classifier.testInstances(Ftt[1])
        if len(nsi)==0:
            nsi=[-np.inf]
        psi=classifier.testBag(Ftt[0]) #max in prion domain   
        
        r=np.max((np.max(nsi),psi))   

        
        xx=testOnSeqs(classifier,HWS,[NRp[k]])[0]   
        
        print "Pos",k,r
#        if np.isnan(r):
        #pdb.set_trace()
        my_ps.append(r)
        my_FH.append(100*np.mean(nsi>psi))   
        my_xx.append(xx)
        
    if(myid!=0):
        comm.send((my_ps,my_FH), dest=0)
    else:
        FH=my_FH
        ps=my_ps        
        for p in range(1,nprocs):
            psp,fhp=comm.recv(source=p)
            FH.extend(fhp)
            ps.extend(psp)            
    
    my_ns=[]
    myN=range((myid*len(N))/nprocs,((myid+1)*len(N))/nprocs)
    my_ns=[]    
    for k in myN:   
        pidx = NRn[k][0].split()[0].upper()
        Ntr=N[:k]+N[k+1:]
        Ntt=N[k]
        classifier = classifierTemplate.__class__(classifierTemplate)
        classifier.train(F,Ntr)    
        out_file = tempfile.NamedTemporaryFile(prefix='yeast_neg_'+pidx+'_',suffix='.cls');
        ofname=getFileParts(out_file.name)[1]+'.cls'; out_file.close();        
        classifier.save(ofname)
        print "Wrote file: ",ofname
        r=classifier.testBag(Ntt) 
        xx=testOnSeqs(classifier,HWS,[NRn[k]])[0]
        print "Neg",k,r
        my_ns.append(r)
        my_xx.append(xx)
    
    if(myid!=0):
        comm.send((my_ns,my_xx), dest=0)
    else:        
        ns=my_ns
        for p in range(1,nprocs):
            nsp,xx=comm.recv(source=p)
            ns.extend(nsp)
            my_xx.extend(xx)        
        print "MYXX",len(my_xx)
        if ofile is not None:
            writePAPAFile(ofile,my_xx)
            print "Output file:",ofile,"Written."
        (_,_,auc)=roc.roc(ps+ns,[+1]*len(ps)+[-1]*len(ns))
        from sklearn.metrics import average_precision_score
        print "AUC-PR",average_precision_score([+1]*len(ps)+[-1]*len(ns), ps+ns)
        print "AUC",auc,"FH",np.mean(FH)        
        return (FH,ps,ns,my_xx)
        
def leaveOneOutCV(classifierTemplate,F,N):
    """
    leave one out cross validation
    """
    ps=[]
    FH=[]
    for k in range(len(F)):
        print "Pos",k
        Ntr=N
        Ftr=F[:k]+F[k+1:]
        Ftt=F[k]
        classifier = classifierTemplate.__class__(classifierTemplate)
        classifier.train(Ftr,Ntr)
        nsi=classifier.testInstances(Ftt[1],norm=True)
        psi=classifier.testBag(Ftt[0],norm=True) #max in prion domain        
        r=np.max((np.max(nsi),psi))    
        print r
        ps.append(r)#np.max((np.max(nsi),psi))
        FH.append(100*np.mean(nsi>psi))        
    ns=[]
    for k in range(len(N)):
        print "Neg",k
        Ftr=F
        Ntr=N[:k]+N[k+1:]
        Ntt=N[k]
        classifier = classifierTemplate.__class__(classifierTemplate)
        classifier.train(Ftr,Ntr)        
        r=classifier.testBag(Ntt,norm=True) 
        print r
        ns.append(r)#np.max((np.max(nsi),psi))
        #ns.append(classifier.testBag(Ntt))
    (_,_,auc)=roc.roc(ps+ns,[+1]*len(ps)+[-1]*len(ns))
    print auc
    
    return (FH,ps,ns)
def holdOutCV(classifierTemplate,F,N,pout=1,nout=1):    
    """
    Hold out "pout" proteins from the positive set and "nout" proteins from the
    negative set and train on the rest and test on the held out set to compute 
    the auc, false hit and true hit ratios
    """
    Ftt_idx=random.sample(range(len(F)),pout)
    Ftr_idx=list(set(range(len(F))).difference(Ftt_idx))
    Ntt_idx=random.sample(range(len(N)),nout)
    Ntr_idx=list(set(range(len(N))).difference(Ntt_idx))
    Ftr=[F[i] for i in Ftr_idx]
    Ntr=[N[i] for i in Ntr_idx]    
    #pdb.set_trace()
    classifier = classifierTemplate.__class__(classifierTemplate)
    classifier.train(Ftr,Ntr)
    
    FH=[]
    ps=[]
    for idx in Ftt_idx:
        nsi=classifier.testInstances(F[idx][1]) #scores of windows in non-prion regions
        psi=classifier.testBag(F[idx][0]) #max in prion domain
        FH.append(100*np.mean(nsi>psi)) #percentage of negative windows that score higher than the most positive window
        ps.append(np.max((np.max(nsi),psi)))
    
    ns=[classifier.testBag(N[i]) for i in Ntt_idx]    
    if len(ns)>0 and len(ps)>0:        
        #(_,_,auc)=roc.roc(ps+ns,[+1]*len(ps)+[-1]*len(ns))
        import sklearn
        auc = sklearn.metrics.roc_auc_score([1]*len(ps)+[-1]*len(ns), ps+ns)
    else:
        auc=np.nan
    TH=100.0*np.mean(np.array(FH)==0.0)
    FH=np.mean(FH)
    return (auc*100,FH,TH)      
    
