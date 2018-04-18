# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:19:06 2014

@author: fayyaz
"""
import add2path
from Bio import SeqIO
from itertools import product
import numpy as np
import random, pdb
from PyML.classifiers.baseClassifiers import Classifier
import matplotlib.pyplot as plt
import PyML
AA='ACDEFGHIKLMNPQRSTVWY'    
from mpi4py import MPI
import os,sys
misvm_path='../Tools/misvm'
sys.path.append(misvm_path)
import misvm
from kidera import *
from papa import *
from myPickle import *
import re
import myPickle
def getWkSpectrum(S,k=1,hws=20):    
    d=len(AA)**k
    sdidx=dict(zip([''.join(a) for a in list(product(*([AA]*k)))],range(d)))
    SF=[]
    for s in S:
        if k>1:
            sq=s[:-k+1]
        else:
            sq=s
        if len(s)-2*hws<=0:
            F=np.zeros((d,1))
        else:
            F=np.zeros((d,len(s)-2*hws))
        
        for i,si in enumerate(sq):
            try:
                if F.shape[1]>1:
                    F[sdidx[s[i:i+k]],np.max((0,i-2*hws)):i+1]+=1
                else:
                    F[sdidx[s[i:i+k]],0]+=1
            except KeyError:
                continue
        SF.append(F)
    return SF
#load sequences for prions
def normalize(X):
    return X/np.sqrt(np.sum(X**2,axis=0))
def loadAnnotatedPrions(ifile='../data/prion_proteins_annotated.fasta'):    
    S=[(record.seq.tostring(),[int(i)-1 for i in record.description.split('#')[1].split('-')]) for record in SeqIO.parse(open(ifile, "rU"), "fasta")]        
    return tuple(map(list,[a for a in zip(*S)]))

class MI1SVM:
    """
    MI-1 SVM (the maximum scoring window of each protein scores higher than the
    negative windows in that protein)
    """
    def __init__(self,arg = None, **args):
        self.W=np.nan
        self.Lambda=1e-3
        self.Epochs=100
        self.History=[]
        if type(arg)==type(self):
            self.Lambda=float(arg.Lambda)
            self.Epochs=long(arg.Epochs)
        
    def train(self,F,Xn=[]):
        Xn=[]
        #N=np.hstack([F[i][1] for i in range(len(F))]+Xn) #Collect all negative windows
        L=self.Lambda     
        W=np.zeros(F[0][0].shape[0])
        sqrtL_inv=1.0/np.sqrt(L)
        norm=np.linalg.norm
        P=range(len(F))
        t=0
        self.History=[]
        for e in range(0,self.Epochs):
            random.shuffle(P)
            for i in P:
                t+=1
                
                eta=1.0/(L*t)
                Vp=np.dot(W,F[i][0]) #most positive positive example
                pi=np.argmax(Vp)
                xp=F[i][0][:,pi]
                
                Vn=np.dot(W,F[i][1]) #most positive negative example of all proteins
                ni=np.argmax(Vn)
                xn=F[i][1][:,ni]
                
                W=(1-L*eta)*W
                if Vp[pi]<(Vn[ni]+1): #if current selection incurs some loss
                   W+=eta*(xp-xn)
                   
                W=np.min((1,(sqrtL_inv)/norm(W)))*W #projection step        
                
            #Compute the primal objective function at the end of each epoch
            #nmax=np.max(np.dot(W,N))
            obj=(L*(norm(W)**2)/2.0)+np.sum([np.max((0,1-(np.max(np.dot(W,f[0]))-np.max(np.dot(W,f[1]))))) for f in F])
            #obj=(L*(norm(W)**2)/2.0)+np.sum([np.max((0,1-(np.max(np.dot(W,f[0]))-nmax))) for f in F])
            self.History.append(obj)
            
        self.W=W
    
    def testInstances(self,x):
        return np.dot(self.W,x)
    
    def testBag(self,f):
        return np.max(np.dot(self.W,f))
        
class MI2SVM:
    """
    Stochastic gradient solver based multiple instance learning SVM that scores
    the most positive instance in a bag to be higher than all negative instances
    in all negative bags
    """
    def __init__(self,arg = None, **args):
        self.W=np.nan
        self.Lambda=1e-3
        self.Epochs=100
        self.History=[]
        if type(arg)==type(self):
            self.Lambda=float(arg.Lambda)
            self.Epochs=long(arg.Epochs)
        
    def train(self,F,Xn=[]):
        N=np.hstack([F[i][1] for i in range(len(F))]+Xn) #Collect all negative windows
        L=self.Lambda     
        W=np.zeros(F[0][0].shape[0])
        sqrtL_inv=1.0/np.sqrt(L)
        norm=np.linalg.norm
        P=range(len(F))
        t=0
        self.History=[]
        W0=W
        obj0=np.inf
        
        for e in range(0,self.Epochs):
            random.shuffle(P)
            for i in P:
                t+=1
                
                eta=1.0/(L*t)
                Vp=np.dot(W,F[i][0]) #most positive positive example
                pi=np.argmax(Vp)
                xp=F[i][0][:,pi]
                
                Vn=np.dot(W,N) #most positive negative example of all proteins
                ni=np.argmax(Vn)
                xn=N[:,ni]
                
                W=(1-L*eta)*W
                if Vp[pi]<(Vn[ni]+1): #if current selection incurs some loss
                   W+=eta*(xp-xn)
                   
                #W=np.min((1,(sqrtL_inv)/norm(W)))*W #projection step        
                
            #Compute the primal objective function at the end of each epoch
            nmax=np.max(np.dot(W,N))
            obj=(L*(norm(W)**2)/2.0)+np.sum([np.max((0,1-(np.max(np.dot(W,f[0]))-nmax))) for f in F])
            if obj<obj0:
                obj0=obj                
                W0=W
                
            self.History.append(obj)
               
        self.W=W0
        sx=[np.max((self.testBag(f[0]),self.testBag(f[1]))) for f in F]+[self.testBag(n) for n in Xn] 
        self.min=np.min(sx) #Max. score of all training examples
        self.max=np.max(sx) #Min score of all training examples
    def testInstances(self,x,norm=False):
        v=np.dot(self.W,x)
        if norm:
            v=2*((v-self.min)/(self.max-self.min))-1            
        return v
    
    def testBag(self,f,norm=False):
        return np.max(self.testInstances(f,norm=norm))#np.max(np.dot(self.W,f))

class classifierEnsemble:
    def __init__(self,classifierTemplate,N=2):   
        
        if not isinstance(classifierTemplate,self.__class__):
            self.classifiers=[classifierTemplate.__class__(classifierTemplate) for i in range(N)]
        else:
            self.classifiers=classifierTemplate.classifiers
    def train(self,F,Xn=[]):
        for i in range(len(self.classifiers)):
            print "Training Classifier",i            
            self.classifiers[i].train(F,Xn)
    def testInstances(self,x,norm=True):
        assert norm
        s=[]
        for i in range(len(self.classifiers)):
            s.append(self.classifiers[i].testInstances(x,norm=norm))
        return np.mean(s,axis=0)
    def testBag(self,f,norm=True):
        return np.max(self.testInstances(f,norm=norm))
        
class MISVM:
    """
    Classical multiple instance learning SVM
    Example: classifierTemplate=MISVM(C=100)
    Scores all instances in all negative bags as negative
    Scores one or more instances in positive bags as positive
    """
    attributes = {'C' : 10.0,
                  'kernel' : 'linear',
                  'scale_C': True,
                  'p' : 2,
                  'gamma' : 1.0,
                  'verbose' : True,
                  'sv_cutoff' : 1e-7,
                  'restarts' : 0,
                  'max_iters': 50}
    def __init__(self,arg=None,**args):  
        for kw in args:        
            if kw in self.attributes:
                self.attributes[kw]=args[kw]
        if type(arg)==type(self):
            self.attributes=arg.attributes
        self.classifier = misvm.MISVM(**self.attributes)
    def train(self,F,Xn=[]):        
        pbags=[f[0].T for f in F]
        nbags=[f[1].T for f in F]+[xn.T for xn in Xn] #bags
        labels=[1.0 for _ in pbags]+[-1.0 for _ in nbags] #bags #bag labels.
        #pdb.set_trace()
        self.classifier.fit(pbags+nbags, labels)
        
    def testBag(self,f):
        if type(f)==type([]):
            bags=[x.T for x in f] 
        else:
            bags=[f.T]
        return self.classifier.predict(bags)
    def testInstances(self,x):
        return self.classifier.predict(x.T)

from PyML import SVM       
class mySVM(SVM):
    """
    My Implementation of SVM
    """
    def __init__(self,arg=None,**args):
        SVM.__init__(self, arg, **args)
    def train(self,F,Xn=[]):
        # Use all examples in f[0] as positive, all in f[1] as negative and all the ones in Xn as negative too
        pset=np.hstack([f[0] for f in F])
        nset=np.hstack(([f[1] for f in F]+Xn))
        X=np.hstack((pset,nset))
        Y=np.array([1]*pset.shape[1]+[-1]*nset.shape[1])
        data=PyML.SparseDataSet(X.T, L = Y)
        SVM.train(self,data)    
    def testInstances(self,x):
        return np.array(SVM.test(self,PyML.SparseDataSet(x.T)).getDecisionFunction())
    def testBag(self,x):
        return np.max(self.testInstances(x))
def leaveOneOutCV(classifierTemplate,F,N):
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
    (_,_,auc)=PyML.evaluators.roc.roc(ps+ns,[+1]*len(ps)+[-1]*len(ns))
    print auc
    pdb.set_trace()
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
        (_,_,auc)=PyML.evaluators.roc.roc(ps+ns,[+1]*len(ps)+[-1]*len(ns))
    else:
        auc=np.nan
    TH=100.0*np.mean(np.array(FH)==0.0)
    FH=np.mean(FH)
    return (auc*100,FH,TH)      
def calc_gini(x): #follow transformed formula
    """Return computed Gini coefficient.
    """
    xsort = sorted(x) # increasing order    
    y = np.cumsum(xsort)    
    B = sum(y) / (y[-1] * len(x))
    return 1 + 1./len(x) - 2*B    
def getGINI(m) :
    return np.atleast_2d([calc_gini(p) for p in m.T])
def interProlineDistance(s,HWS=20):
    WS=float(2*HWS+1)
    #d=np.array([np.mean(np.diff(np.nonzero(w=='P')[0]))/WS for w in rolling_window(np.array([a for a in s]),WS)[:]] )
    D=[np.diff(np.nonzero(w=='P')[0]) for w in rolling_window(np.array([a for a in s]),WS)[:]]
    mD=[]    
    for d in D:        
        if len(d):
            m=np.min(d)#(np.min(d),np.max(d))
        else:
            m=0#(0,0)
        mD.append(m)
    return np.array(mD).T
    
def propertify(seq,aa_dict,HWS=20):
    v=np.mean(aa_dict.values())
    p=np.zeros(len(seq))
    p.fill(v)
    for i,a in enumerate(seq):
        try:
            p[i]=aa_dict[a]
        except KeyError:
            continue
    return np.mean(rolling_window(p,2*HWS+1),axis=1)
    
def foldIndex(seq,HWS=20):
    return 2.785*propertify(seq,hydrophobicity,HWS=HWS)-np.abs(propertify(seq,charge,HWS=HWS))-1.151
    
def papaIndex(seq,HWS=20):    
    P = propertify(seq,propensities,HWS=HWS)
    #P[foldIndex(seq,HWS=HWS)>0]=-1.0 # if folded then the region cannot be a prion
    return P
    
def featureExtraction(S,HWS=20):    
    K=1
    PSK=[normalize(ps) for ps in getWkSpectrum(S,k=K,hws=HWS)] #K spectrum features
    PSD=[interProlineDistance(s,HWS=HWS) for s in S]
    #PSG=[getGINI(p) for p in PSK]
    #PSF=[foldIndex(s,HWS=HWS) for s in S]
    PSP=[papaIndex(s,HWS=HWS) for s in S]
    
    PS=[normalize(np.vstack(p)) for p in zip(PSK,PSD,PSP)]
    #PSM=[ps for ps in avgdProperties(S,HWS=HWS)] #Properties averages
    #PSS=[ps for ps in stdProperties(S,HWS=HWS)] #Properties averages    
    return PS
def testFasta(classifier,HWS,fname,ofile=None):    
    if type(fname)==type((0,)):
        NRF=fname[1]
        fname=fname[0]
    NR=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(fname, "rU"), "fasta")]    
    NR2idx=[i for i,(h,n) in enumerate(NR) if len(n)>2*HWS+1]   
    NR2=[NR[i][1] for i in NR2idx]
    HNR2=np.array([NR[i][0] for i in NR2idx])
    try:
        NRF
    except NameError:
        NRF=featureExtraction(NR2,HWS=HWS)
    for i in range(len(NRF)):
        NRF[i][np.isnan(NRF[i])]=0.0
    yN=np.array([classifier.testBag(f) for f in NRF])
    yidx=np.argsort(-yN)
    yN=yN[yidx]
    HNR2=HNR2[yidx]
    R=zip(HNR2,yN)
    if ofile is not None:
        with open(ofile,'w') as fout:
            for (hdr,score) in R:
                pos=-1 #position to be added later
                fout.write(str(hdr) + ',' + str(score) + ',' + str(pos) + '\n') 
    return R
def readPAPAFile(fname,asTuple=False):
    S=[]    
    import re
    with open(fname,'r') as f:
        for l in f:
            mm=re.search(r"\,([-]*[0-9]+\.[0-9]+)\,", l)
            if mm is not None:
                try:
                    v=float(mm.group(1))
                    if asTuple:
                        S.append((l.split(',')[0].split()[0].lower(),v))
                    else:
                        S.append(v)                    
                except Exception as exc:
                    print exc                                     
                    continue
    
    return S    
if __name__=="__main__":
    HWS=20
    	
    Ntrials=8;pout=9;nout=18
    
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()  

    nfile='../data/nonprions.fasta'
    S,A=loadAnnotatedPrions(ifile='../data/prion_proteins_annotated_18.fasta')#')yhf
    
    #A=[[np.max((0,a-(HWS))),b] for (a,b) in A]
    
    PS=featureExtraction(S,HWS=HWS)
    
    NS=[record.seq.tostring() for record in SeqIO.parse(open(nfile, "rU"), "fasta")]    
    #[np.vstack((normalize(a),normalize(b))) for (a,b) in zip(getWkSpectrum(NS,k=1),getWkSpectrum(NS,k=2))]#getWkSpectrum(NS)#
    Xn=featureExtraction(NS,HWS=HWS)
    #separate the positive windows from the negative ones 
    F=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-HWS))],ps[:,a[1]-HWS:]))) for ps,a in zip(PS,A)]
#    classifierTemplate=MISVM(C=100)
#    classifierTemplate=mySVM(C=100)
    
    classifierTemplate=MI2SVM()
    classifierTemplate.Lambda=5e-6
    classifierTemplate.Epochs=5000
    
#    classifier=classifierEnsemble(classifierTemplate,N=5)
    #classifier=classifierTemplate.__class__(classifierTemplate)
#    classifier.train(F,Xn)  
#    testFasta(classifier,HWS,'../data/human_fly.fasta',ofile='human_fly.txt')
#    testFasta(classifier,HWS,'../data/prion_proteins_annotated_18.fasta',ofile='pos_18.txt')
#    testFasta(classifier,HWS,nfile,ofile='neg_18.txt')
#    testFasta(classifier,HWS,'../data/prion_proteins_annotated_4.fasta',ofile='pos_4.txt')
#    
#    testFasta(classifier,HWS,'..\data\AlbSequences_others.fasta',ofile='alberti_others.txt')
#    1/0    
    """
    testFasta(classifier,HWS,'..\data\disprot_v6.02.fasta',ofile='disprot.txt')
    NRF=myPickle.load('NRF.mkl')
    testFasta(classifier,HWS,('../data/pdbaanr/pdbaa.nr',NRF),ofile='neg_pdb.txt')

    yP=np.array(readPAPAFile('human_fly.txt'))
    yN=np.array(readPAPAFile('neg_pdb.txt'))
    print np.sum(yN>np.min(yP))
    plt.plot(np.sort(yP),([np.sum(yN>=np.sort(yP)[i]) for i in range(len(yP))]),'o-');
    plt.plot(np.sort(yP),[np.sum(yP<np.sort(yP)[i]) for i in range(len(yP))],'.-');
    plt.xlabel('Threshold');plt.ylabel('Count');
    plt.legend(['Predicted positives from NR-PDB (Total: 62,314)','Predicted Negatives from Prion proteins (Total: 18)']);
    plt.grid();plt.show()
    """
    
#    
        
############PROCESSING FOR NR-PDB LOPO
#    NRF=featureExtraction(NR2,HWS=HWS)
#    yN=[classifier.testBag(f) for f in NRF]        
#    
#    import yard
#    pdbfile='../data/pdbaanr/pdbaa.nr'
#    NR=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(pdbfile, "rU"), "fasta")]
#    
#    NR2idx=[i for i,(h,n) in enumerate(NR) if len(n)>2*HWS+1]   
#    NR2=[NR[i][1] for i in NR2idx]
#    HNR2=[NR[i][0] for i in NR2idx]
#    
#    #NRF=featureExtraction(NR2,HWS=HWS)
#    #dump('NRF.mkl',NRF)
#    NRF=load('NRF.mkl')
#    for i in range(len(NRF)):
#        NRF[i][np.isnan(NRF[i])]=0.0
#    
#    PR=[]
#    ROC=[]
#    SX=[]
#    for i in range(len(F)):
#        Fin=F[:i]+F[i+1:]
#        classifier=classifierTemplate.__class__(classifierTemplate)
#        classifier.train(Fin,Xn)    
#        yN=[classifier.testBag(f) for f in NRF]        
#        yP=[np.max((classifier.testBag(F[i][0]),classifier.testBag(F[i][1])))]
#        yL=np.array([+1.0]*len(yP)+[-1.0]*len(yN))
#        yV=np.array(yP+yN)
#        zpr=yard.PrecisionRecallCurve(yard.BinaryClassifierData(zip(yV,yL)))#PrecisionRecallCurve
#        zroc=yard.ROCCurve(yard.BinaryClassifierData(zip(yV,yL)))
#        s=100*np.mean(yV[1:]>yV[0])
#        print i,s
#        SX.append(s)    
#        PR.append(zpr)
#        ROC.append(zroc)
#    print "DONE", np.mean(SX),np.std(SX),np.mean([z.auc() for z in ROC]),np.std([z.auc() for z in ROC])
#    
###################################
###################################
#PROCESSING AFTER TRAINING ON ALL DATA AND TEST ON NR-PDB
######    
#    classifier=classifierTemplate.__class__(classifierTemplate)
#    classifier.train(F,Xn)    
#    yN=[classifier.testBag(f) for f in NRF]
#    yP=[np.max((classifier.testBag(F[i][0]),classifier.testBag(F[i][1]))) for i in range(len(F))]
#    yL=np.array([+1.0]*len(yP)+[-1.0]*len(yN))
#    yV=np.array(yP+yN)    
#    np.savetxt('vals.txt',yV)
#    np.savetxt('lbls.txt',yL)
#    yV=np.loadtxt('vals.txt')    
#    yL=np.loadtxt('lbls.txt')
#    nidx=np.where(yL==-1)[0];pidx=np.where(yL==1)[0]
#    lidx=np.where(yV[nidx]>np.min(yV[pidx]))[0]
#    gidx=lidx[np.argsort(-yV[nidx[lidx]])]
#    print ['>'+HNR2[i]+'\n'+NR2[i] for i in gidx] #Save to file through notepad
#    tpdbfile='top_nrpdb.fasta'
#    NR=[(record.description,record.seq.tostring()) for record in SeqIO.parse(open(tpdbfile, "rU"), "fasta")]
#    NR2idx=[i for i,(h,n) in enumerate(NR) if len(n)>2*HWS+1]   
#    NR2=[NR[i][1] for i in NR2idx]
#    HNR2=[NR[i][0] for i in NR2idx]
#    plt.plot(np.sort(yV[pidx]),[np.sum(yV[nidx]>=np.sort(yV[pidx])[i]) for i in range(len(pidx))],'o-');plt.plot(np.sort(yV[pidx]),[np.sum(yV[pidx]<np.sort(yV[pidx])[i]) for i in range(len(pidx))],'.-');plt.xlabel('Threshold');plt.ylabel('Count');plt.legend(['Predicted positives from NR-PDB (Total: 62,314)','Predicted Negatives from Prion proteins (Total: 18)']);plt.grid();plt.show()
###############################    
#    W=classifier.W
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    rects1 = ax.bar(np.arange(len(W))   , W, 0.8  )
#    ax.set_ylabel('Weight values')
#    plt.xticks(np.arange(len(AA))+0.5,[a for a in AA])
#    plt.grid()
#    plt.show()
    
    
#    plt.plot([np.max(classifier.testInstances(PS[idx])) for idx in range(len(PS))]); plt.plot([np.max(classifier.testInstances(Xn[idx])) for idx in range(len(Xn))]);plt.show()
    
#    St,At=loadAnnotatedPrions(ifile='../data/prion_proteins_annotated_4.fasta')
#    PSt=featureExtraction(St,HWS=HWS)
#    plt.plot([0.1]*len(PS),[classifier.testBag(p) for p in PS],'ro');plt.plot([0.2]*len(Xn),[classifier.testBag(p) for p in Xn],'bs');plt.plot([0.3]*len(PSt),[classifier.testBag(p) for p in PSt],'k^');plt.axis([0,0.4,-10,+10]);plt.show()
#    PSD=[interProlineDistance(s,HWS=20) for s in St]
#    PSG=[getGINI(p) for p in PSK]
#    #PSM=[ps for ps in avgdProperties(S,HWS=HWS)] #Properties averages
#    #PSS=[ps for ps in stdProperties(S,HWS=HWS)] #Properties averages
#    PSt=[np.vstack(p) for p in zip(PSK)]  
#    Ft=[(ps[:,np.max((0,a[0]-HWS)):a[1]-HWS],np.hstack((ps[:,0:np.max((0,a[0]-HWS))],ps[:,a[1]-HWS:]))) for ps,a in zip(PSt,At)]
#    #idx=2;plt.plot(classifier.testInstances(Ft[idx][0]));plt.plot(classifier.testInstances(Ft[idx][1]));plt.show()
#    idx=1;plt.plot(classifier.testInstances(PSt[idx]));plt.plot(range(At[idx][0],np.min((At[idx][1],PSt[idx].shape[1]))),classifier.testInstances(PSt[idx][:,At[idx][0]:At[idx][1]]));plt.show()
#    idx=5;plt.plot(classifier.testInstances(PS[idx]));plt.plot(range(A[idx][0],np.min((A[idx][1],PS[idx].shape[1]))),classifier.testInstances(PS[idx][:,A[idx][0]:A[idx][1]]));plt.show()
#    
#    1/0
############### PROTOCOL ############    
    cE=classifierEnsemble(classifierTemplate,N=5)
    #leaveOneOutCV(cE,F,Xn)
    
    
    R=[]
    if myid>0:
        niters=np.floor(Ntrials/nprocs)
    else:
        niters=Ntrials-(nprocs-1)*np.floor(Ntrials/nprocs)
    print "My id is",myid,"and I am processing",niters
    for _ in range(int(niters)):
        r=holdOutCV(cE,F,Xn,pout=pout,nout=nout)
        print r
        R.append(r)
    if myid>0:
        comm.send(R,dest=0)
    else:
        for p in range(1,nprocs):
            R.extend(comm.recv(source=p))
        print "RESULTS",np.mean(R,axis=0),np.std(R,axis=0)
    
#############################    
#    R=stratifiedCV(classifierTemplate,F,Xn,nfolds=18,comm=comm,myid=myid,nprocs=nprocs)
#    if myid==0:
#        print R
    #FH=baggedCV(classifierTemplate,F,nfolds=22)
    """
classifier=classifierTemplate.__class__(classifierTemplate)
classifier.train(F)
W=classifier.W
    
    #Ps=[np.max((classifier.testBag(F[i][0]),classifier.testBag(F[i][1]))) for i in range(len(F))]
    Ps=[classifier.testBag(F[i][0]) for i in range(len(F))]
    Ns=[classifier.testBag(Xn[i]) for i in range(len(Xn))]
    
    (fpr,tpr,auc)=PyML.evaluators.roc.roc(Ps+Ns,[+1]*len(Ps)+[-1]*len(Ns))
    
    print auc
    plt.plot(fpr,tpr);plt.show()
    
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(np.arange(len(W))   , W, 0.8  )
ax.set_ylabel('Weight values')
plt.xticks(np.arange(len(AA))+0.5,[a for a in AA])
plt.grid()
plt.show()
       #print L*np.dot(W,W)/2.0+np.sum([np.max((0,1-(np.max(np.dot(W,f[0]))-np.max(np.dot(W,f[1]))))) for f in F])
    """