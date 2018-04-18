# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:43:16 2014

@author: root
"""

#import PyML,pdb
import numpy as np
from Bio import SeqIO
from itertools import product
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.solvers import qp
#from PyML.classifiers.baseClassifiers import Classifier
#from propensity_classifier import PropensityClassifier
#from PyML.classifiers import modelSelection
import random
import os,sys
misvm_path='../Tools/misvm'
sys.path.append(misvm_path)
import misvm
AA='ACDEFGHIKLMNPQRSTVWY'    
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
#NS=PyML.sequenceData.fasta_read(nfile) #Adds '\r'
#PS=PyML.sequenceData.fasta_read(pfile)
class MISVM (Classifier) :
    attributes = {'C' : 1000.0}
    def __init__(self, arg = None, **args) :
        Classifier.__init__(self, **args)    
        if type(arg)==type(self):
            self.C=arg.C
    def train(self, data, **args) :
        d=len(data[0]) #dimensions of the feature vector
        self.d=d
        #pdb.set_trace()
        num_neg_wins = data.labels.classSize[0] #number of negative examples
        num_pos_wins = data.labels.classSize[1] #number of positive examples
        num_examples = num_neg_wins + num_pos_wins 
        B=data.labels.patternID #Bag id of each example
        Y=data.labels.L #Label of each example
        pid_pos = np.unique([p for (p,y) in zip(B,Y) if y>0]) #list of bag ids of positive examples
        pid_neg = np.unique([p for (p,y) in zip(B,Y) if y<0]) #list of bag ids of negative examples
        num_pos = len(pid_pos) #number of positive proteins (bags)
        num_neg = len(pid_neg) #number of negative proteins (bags)
        
        X = data.getMatrix() #Data (num_examples x d)
        
        Pidx=[] #List of indices of examples of each positive bag in correspondence with pid_pos
        S=np.zeros(num_pos,np.int) #index in X of representative example of each positive bag
        Gn1=[]
        pX=np.zeros((num_pos,d)) #initialization of the 
        for i,p in enumerate(pid_pos):
            Pidx.append(np.nonzero(B==p)[0])
            S[i]=Pidx[i][0] #initialization
            pX[i,:]=np.mean(X[Pidx[i],:],axis=0)
            z=np.zeros((1,num_pos+num_neg))
            z[:,i]=-1
            Gn1.extend(z)
        
        Nidx=[]    
        for i,n in enumerate(pid_neg):
            nidx=np.nonzero(B==n)[0]
            Nidx.extend(nidx)
            z=np.zeros((len(nidx),num_pos+num_neg))
            z[:,num_pos+i]=-1
            Gn1.extend(z)
        Gn1=np.array(Gn1)
        Gn1=np.vstack((Gn1,-np.eye(num_pos+num_neg)))
        Gnb=np.vstack((-np.ones((num_pos,1)),np.ones((num_neg_wins,1)),np.zeros((num_pos+num_neg,1))))
        Gnx=np.vstack((-pX,X[Nidx,:],np.zeros((num_pos+num_neg,d))))
        G=np.hstack((Gnx,Gnb,Gn1))
        h=np.vstack((-np.ones((num_pos+num_neg_wins,1)),np.zeros((num_pos+num_neg,1))))
        P=np.eye(d+1+num_pos+num_neg)
        P[d:,d:]=0
        self.C=np.double(self.C)        
        Cp=self.C/num_pos
        Cn=self.C/num_neg
        
        q=np.vstack((np.zeros((d+1,1)),Cp*np.ones((num_pos,1)),Cn*np.ones((num_neg,1))))
        for niter in range(10):
            s0=qp(matrix(P), matrix(q), matrix(G), matrix(h))    
            self.s0=s0
            self.w=np.array(s0['x'][:d])
            self.b=np.double(s0['x'][d])
            S0=S+0
            for i,p in enumerate(pid_pos):
                S[i]=Pidx[i][np.argmax(np.dot(X[Pidx[i],:],self.w))] 
            if np.all(S0==S):    
                print "############# CONVERGED in %d iterations ############" %niter
                break
            else:
                print "############# CHANGED %d ############" % np.sum(S0!=S)
            G[:num_pos,:d]=-X[S,:]
#        nw=np.vstack((self.w,self.b))/np.linalg.norm(np.vstack((self.w,self.b)))
#        self.w=nw[:-1]
#        self.b=nw[-1]
    def decisionFunc(self, data, i) :        
        return np.double(np.dot(data[i],self.w)+self.b)
        
    def classify(self, data, i) :
        return self.twoClassClassify(data, i)
        
def baggedCV(classifierTemplate,data,nfolds=10):
    bag_idx=data.labels.patternID
    bag_ids=np.asarray(np.unique(bag_idx),np.int32)
    Z=np.zeros(len(bag_idx))
    Y=np.array(data.labels.L)
    C=[]
    for tt_bags in chunk(bag_ids,nfolds):
        tr_idx=[i for (i,b) in enumerate(bag_idx) if b not in tt_bags]
        tt_idx=[i for (i,b) in enumerate(bag_idx) if b in tt_bags]
        classifier = classifierTemplate.__class__(classifierTemplate)
        classifier.train(data.__class__(data, patterns = tr_idx))
        r=classifier.test(data.__class__(data, patterns = tt_idx))
        Z[tt_idx]=r.getDecisionFunction()
        C.append(classifier)
    Zbag=[]
    Ybag=[]
    
    for bag in bag_ids:
        idx=[int(i) for i in np.nonzero(bag_idx==bag)[0]]
        
        Zbag.append(np.max(Z[idx]))
        Ybag.append(np.max(Y[idx]))
       
    return Zbag,Ybag,C
def bagCV(bags,labels,nfolds=10):    
    
    Zbag=[]
    Ybag=[]
    C=[]
    
    for tt_idx in chunk(range(len(bags)),nfolds):
        classifier = misvm.MISVM(kernel='linear', C=100)
        tr_bags,tr_labels=map(list,zip(*[(b,l) for (i,(b,l)) in enumerate(zip(bags,labels)) if i not in tt_idx]))
        tt_bags,tt_labels=map(list,zip(*[(b,l) for (i,(b,l)) in enumerate(zip(bags,labels)) if i in tt_idx]))
        classifier.fit(tr_bags, tr_labels)
        zpi=classifier.predict(tt_bags)
        Zbag.extend(zpi)
        Ybag.extend(tt_labels)
        C.append(classifier)
    
    return Zbag,Ybag,C
def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in xrange(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra    
def normalize(X):
    return X/np.sqrt(np.sum(X**2,axis=0))
if __name__=="__main__":
    
    pfile='../data/prions.fasta'
    nfile='../data/nonprions.fasta'
    PS=[record.seq.tostring() for record in SeqIO.parse(open(pfile, "rU"), "fasta")]
    NS=[record.seq.tostring() for record in SeqIO.parse(open(nfile, "rU"), "fasta")]
    1/0
    """
    from readAlberti import loadAlberti
    D=loadAlberti()
    PS=[D[id][1] for id in D if np.sum(np.array(D[id][0])>0)==4]
    NS=[D[id][1] for id in D if np.sum(np.array(D[id][0])>0)==0]
    """
    PSF=getWkSpectrum(PS,k=2)#[np.vstack((normalize(a),normalize(b))) for (a,b) in zip(getWkSpectrum(PS,k=1),getWkSpectrum(PS,k=2))]#
    Xp=np.hstack(PSF)
    NSF=getWkSpectrum(NS,k=2)#[np.vstack((normalize(a),normalize(b))) for (a,b) in zip(getWkSpectrum(NS,k=1),getWkSpectrum(NS,k=2))]#getWkSpectrum(NS)#
    Xn=np.hstack(NSF)
    X=normalize(np.hstack((Xp,Xn)))
    #X=X/np.sqrt(np.sum(X**2,axis=0))
    Np=Xp.shape[1] #Number of windows in positive examples
    Y=np.zeros(X.shape[1])
    Y[:Np]=1
    Y[Np:]=-1
    P=np.zeros(Y.shape) #bag id for each window
    i0=0
    for i,sf in enumerate(PSF+NSF):
        i1=i0+sf.shape[1]
        P[i0:i1]=i
        i0=i1
    data=PyML.SparseDataSet(X.T, L = Y,patternID=P)
    bags=[psf.T for psf in PSF]+[nsf.T for nsf in NSF] #bags
    labels=[+1 for psf in PSF]+[-1 for nsf in NSF] #bag labels
    Clist=[0.05]#
    nfolds=2
    NX=0
    prot_ids=np.asarray(np.unique(data.labels.patternID),np.int32)
    fsize=int(np.ceil((len(prot_ids)-1)/float(nfolds)))
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    
    ncvf=len(prot_ids)
    Zp=[]
    Yp=[]
    Cout=[]
    ntrials=100
    A=[]
    
    classifierTemplate=PyML.SVM(C=1.0)#PyML.SVM
    classifiers=[]
    for trial in range(ntrials):
        #Z,Y,C=baggedCV(classifierTemplate,data)
        Z,Y,C=bagCV(bags,labels,nfolds=2)
        (_,_,auc)=roc.roc(list(Z),list(Y));
        (_,_,auc_0p1)=roc.roc(list(Z),list(Y),rocN=0.2);
        acc=np.mean((2*(np.array(Z)>0)-1)==Y)
        #plt.plot(fpr,tpr);plt.title('AUC='+str(auc));plt.grid();plt.axis([0,1,0,1]);plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
        A.append([auc,acc,auc_0p1])
        classifiers.extend(C)
    print "Accuracy:",np.mean(A,axis=0),np.std(A,axis=0)
    """
    try:
        Ws=np.hstack([np.atleast_2d(c.model.w).T for c in classifiers])
    except:
        Ws=np.hstack([c.w for c in classifiers]) 
        
    W=np.mean(Ws,axis=1)
    """
    #W=np.zeros(len(data[0]))
    """
    classifierTemplate=MISVM#PyML.SVM#
    W=np.zeros((len(data[0])+1,len(prot_ids)))
    for p in prot_ids: #Leave one protein out cross validation
        ttidx=[int(i) for i in np.nonzero(data.labels.patternID==p)[0]] #test index
        tridx=[int(i) for i in np.nonzero(data.labels.patternID!=p)[0]] #training indieces
        Cscore=np.zeros(len(Clist)) #Store results for each C-value
        
        for nxiter in range(NX):
            iCscore=[]
            p_=list(set(prot_ids).difference([p]))
            random.shuffle(p_)
            pfolds=lol(p_,fsize)
            ittidx=[[] for _ in range(nfolds)]
            itridx=[[] for _ in range(nfolds)]
            for f in range(nfolds):
                ittidx[f]=[i for i,bid in enumerate(data.labels.patternID) if bid in pfolds[f]]
                itridx[f]=[i for i,bid in enumerate(data.labels.patternID) if bid not in pfolds[f]+[p]]
            
            for c in Clist:        
                s=classifierTemplate(C=c)        
                r0=PyML.evaluators.assess.cvFromFolds(s,data,itridx,ittidx)
                Z=r0.getDecisionFunction()
                D={}
                for f in range(nfolds):
                    bidx=data.labels.patternID[ittidx[f]]
                    yf=np.array(data.labels.L)[ittidx[f]]
                    Zf=np.array(Z[f])
                    for pf in pfolds[f]:
                        D[pf]=(np.max(Zf[bidx==pf]),np.max(yf[bidx==pf]))
                D=np.array(D.values())
                
                (_,_,auc)=PyML.evaluators.roc.roc(list(D[:,0]),list(D[:,1]))   
                acc=np.mean(2*(D[:,0]>0)-1==D[:,1])
                iCscore.append(auc)
            Cscore+=iCscore
        Cscore/=NX
    #    classifier = misvm.MISVM(kernel='linear', C=Clist[np.argmax(Cscore)])
    #    classifier.fit(bags[:p]+bags[p+1:], labels[:p]+labels[p+1:])
    #    zpi=classifier.predict(bags[p])
        classifier=classifierTemplate(C=Clist[np.argmax(Cscore)])    
        classifier.train(data.__class__(data, patterns = tridx))
        zpi=classifier.test(data.__class__(data, patterns = ttidx)).getDecisionFunction()
        Zp.append(np.max(zpi))
        Yp.append(np.max(Y[ttidx]))
        Cout.append(classifier.C)
    #    W[:-1,p]=classifier.w.flatten()
    #    W[-1,p]=classifier.b
    
    Yp=np.array(Yp)
    Zp=np.array(Zp)
    Lp=2*(Zp>0)-1
    print "Accuracy: ",np.mean((2*(Zp>0)-1)==Yp),(np.mean(Lp[Yp==1]==1)+np.mean(Lp[Yp==-1]==-1))/2.0
    (fpr,tpr,auc)=PyML.evaluators.roc.roc(list(Zp),list(Yp));plt.plot(fpr,tpr);plt.title('AUC='+str(auc));plt.grid();plt.axis([0,1,0,1]);plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
    """
    """
    Z=np.zeros(Y.shape)
    Z.fill(np.nan)
    Zp=np.zeros(ncvf)
    Yp=np.zeros(ncvf)
    for f in np.unique(data.labels.patternID):
        print f
        s=MISVM()
        ttidx=[int(i) for i in np.nonzero(data.labels.patternID==f)[0]]
        tridx=[int(i) for i in np.nonzero(data.labels.patternID!=f)[0]]
        #r=s.trainTest(data,tridx,ttidx)
        Z[ttidx]=np.dot(W,X[:,ttidx])#r.getDecisionFunction()
        Zp[f]=np.max(Z[ttidx])
        Yp[f]=np.max(Y[ttidx])
        print Zp[f]
        #pdb.set_trace()
    (fpr,tpr,auc)=PyML.evaluators.roc.roc(list(Zp),list(Yp));plt.plot(fpr,tpr);plt.title('AUC='+str(auc));plt.grid();plt.axis([0,1,0,1]);plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
    """
    """    
    #Z=r.getDecisionFunction()
    Z=np.dot(data.getMatrix(),W)
    Zp=np.zeros(ncvf)
    Yp=np.zeros(ncvf)
    for f in range(ncvf):
        ttidx=[int(i) for i in np.nonzero(data.labels.patternID==f)[0]]
        Zp[f]=np.max(Z[ttidx])
        Yp[f]=np.max(Y[ttidx])
    
    (fpr,tpr,auc)=PyML.evaluators.roc.roc(list(Zp),list(Yp));plt.plot(fpr,tpr);plt.title('AUC='+str(auc));plt.grid();plt.axis([0,1,0,1]);plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
    """
    """
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(np.arange(len(W))   , W, 0.8  )
ax.set_ylabel('Weight values')
plt.xticks(np.arange(len(AA))+0.5,[a for a in AA])
plt.grid()
plt.show()
    """
    
    """
    W=s.w
k=2
d=len(AA)**k
sdidx_inv=dict(zip(range(d),[''.join(a) for a in list(product(*([AA]*k)))]))
sdidx=dict(zip([''.join(a) for a in list(product(*([AA]*k)))],range(d)))
Wf=[]
for i,w in enumerate(W):
    Wf.append((sdidx_inv[i],w))
Wf=dict(Wf) 
Wx=np.zeros((len(AA),len(AA)))
for i,a in enumerate(AA):
    for j,b in enumerate(AA):
        Wx[i,j]=Wf[a+b]
plt.imshow(Wx,interpolation="nearest");plt.colorbar();plt.xticks(range(len(AA)),[a for a in AA]);plt.yticks(range(len(AA)),[a for a in AA]);plt.show()
    """