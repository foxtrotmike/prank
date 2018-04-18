# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:04:45 2014
Implementation of MIX-SVM
@author: fayyaz
"""
import numpy as np
import random
from baseClassifier import Classifier
class MI2SVM(Classifier):
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
        if isinstance(arg,self.__class__):
            self.Lambda=float(arg.Lambda)
            self.Epochs=long(arg.Epochs)
        
    def train(self,F,Xn=[]):
        Fnegs=np.hstack([F[i][1] for i in range(len(F))])
        N=np.hstack((Fnegs,np.hstack(Xn))) #Collect all negative windows
        Fnegs_idx=Fnegs.shape[1]
        
        L=self.Lambda     
        #import pdb; pdb.set_trace()
        W=np.zeros(F[0][0].shape[0])
        sqrtL_inv=1.0/np.sqrt(L)
        norm=np.linalg.norm
        P=range(len(F))
        t=0
        self.History=[]
        W0=W
        obj0=np.inf
        #import pdb; pdb.set_trace()
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
                vni = Vn[ni]
                #ni=np.random.choice(len(N))
                #vni = np.dot(W,N[:,ni])
                xn=N[:,ni]

                if ni>Fnegs_idx: #if current selection incurs some loss
                   peta = 1.0
                else: #if current selection incurs some loss
                   peta = 0.9
                #peta=((ni>=Fnegs_idx)+1.0)*0.5                


                
                W=(1-L*eta)*W
                
                if Vp[pi]<(vni+1): #if current selection incurs some loss
                   W+=eta*peta*(xp-xn)
                   
                #W=np.min((1,(sqrtL_inv)/norm(W)))*W #projection step                        
            #Compute the primal objective function at the end of each epoch
            
#            nmax=np.max(np.dot(W,N))
#            
#            obj=(L*(norm(W)**2)/2.0)+np.sum([np.max((0,1-(np.max(np.dot(W,f[0]))-nmax))) for f in F])
#            if obj<obj0:
#                obj0=obj                
#                W0=W
                
            #self.History.append(obj)
            
            W0=W
        self.W=W0
        self.b=0
        
        sxp=[np.max((self.testBag(f[0]),self.testBag(f[1]))) for f in F]
        sxn=[self.testBag(n) for n in Xn] 
        
#        mn=np.min(sx) #Max. score of all training examples
#        mx=np.max(sx) #Min score of all training examples
        mux = (np.mean(sxp)+np.mean(sxn))/2.0
        sux = (np.std(sxp)+np.std(sxn))/2.0
        self.W = self.W/sux
        self.b = -mux/sux
#        self.W=(2.0*self.W)/(mx-mn)
#        self.b=-(2.0*mn/(mx-mn))-1
        
    def testInstances(self,x):
        v=np.dot(self.W,x)+self.b   
        return v
    
#    def testBag(self,f,pos=False,vec=False):
#        v=self.testInstances(f)        
#        i=np.argmax(v)        
#        if pos and vec:
#            return v[i],i,list(v)
#        elif pos:
#            return v[i],i
#        else:
#            return v[i]
#        v=self.testInstances(f,norm=norm)        
#        i=np.argmax(v)        
#        if pos:
#            return v[i],i
#        else:
#            return v[i]
    def toString(self):
        import json
        s='#Name='+str(self.__class__)
        s+='#Lambda='+str(self.Lambda)
        s+='#Epochs='+str(self.Epochs)
        s+='#W='+str(json.dumps(self.W.tolist()))
        s+='#b='+str(self.b)
        return s
        
    def fromString(self,s):    
        import json
        for token in s.split('#'):
            if token.find('W=')>=0:
                self.W=np.array(json.loads(token.split('=')[1]))
            elif token.find('b=')>=0:
                self.b=float(token.split('=')[1])   
            elif token.find('Lambda=')>=0:
                self.Lambda=float(token.split('=')[1])
            elif token.find('Epochs=')>=0:
                self.Epochs=long(token.split('=')[1])
#            else:
#                raise ValueError("Unknown token encountered"+token)
    
    def merge(self,flist):
        import json
        
        if type(flist)==type(''):
            flist=[flist]
        S=[]
        for f in flist:
            with open(f,'r') as fin:
                S.extend(json.loads(fin.read()))
        W=[]
        b=[]
        for s in S:
            self.fromString(s)
            W.append(self.W)
            b.append(self.b)
        #import pdb;pdb.set_trace()
        self.W=np.mean(W,axis=0)
        self.b=np.mean(b)
        
            