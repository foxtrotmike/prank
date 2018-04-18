# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:03:09 2014

@author: fayyaz
"""
from itertools import product
from kidera import *
from papa import *
import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'    
from scipy.sparse import csr_matrix
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

def normalize(X):
    return X/(1e-10+np.sqrt(np.sum(X**2,axis=0)))
    
def interResidueDistance(s,HWS=20,res='P'):    
    WS=float(2*HWS+1)
    #d=np.array([np.mean(np.diff(np.nonzero(w=='P')[0]))/WS for w in rolling_window(np.array([a for a in s]),WS)[:]] )
    D=[np.diff(np.nonzero(w==res)[0]) for w in rolling_window(np.array([a for a in s]),WS)[:]]
    mD=[]    
    for d in D:        
        if len(d):
            m=np.min(d)#(np.min(d),np.max(d))
        else:
            m=0#(0,0)
        mD.append(m)
    return np.array(mD).T/WS
    
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
    for i in range(len(S)):
        S[i] = ''.join(si for si in S[i] if si in AA)
    PSK=[normalize(ps) for ps in getWkSpectrum(S,k=K,hws=HWS)] #K spectrum features
#    PSD=[interResidueDistance(s,HWS=HWS) for s in S] #inter-proline distance only
    
    #PSD=zip(*[[interResidueDistance(s,HWS=HWS,res=a) for s in S] for a in AA])
    
    #PSG=[getGINI(p) for p in PSK]
    #PSF=[foldIndex(s,HWS=HWS) for s in S]
    
#    PSP=[papaIndex(s,HWS=HWS) for s in S] #range of 0 to 1.0
    
    PS=[np.vstack(p) for p in zip(PSK)]#,PSD,PSP
    
    #PSM=[ps for ps in avgdProperties(S,HWS=HWS)] #Properties averages
    #PSS=[ps for ps in stdProperties(S,HWS=HWS)] #Properties averages  
    
#    from spectrum import PDSpectrumizer
#    sps = PDSpectrumizer(k = K, hws = HWS)
#    PSP = [sps.spectrumize(s).T.toarray() for s in S]    
#    import pdb; pdb.set_trace()
    return PS    