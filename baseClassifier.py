# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:10:59 2014

@author: fayyaz
"""
import numpy as np
class Classifier:
    def __init__(self,arg=None,**args):
        pass
    def train(self,F,Xn=[]):
        pass
    def testInstances(self,x):
        pass
    def testBag(self,f,pos=False,vec=False):       
        
        v = self.testInstances(f)     
        if len(v)==0:
            v=[-np.inf]
        
        i=np.argmax(v)  
        vmax=v[i]
         
        if pos and vec:
            return vmax,i,list(v)
        elif pos:
            return vmax,i
        else:
            return vmax
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    def fromString(self,s):
        pass
    def toString(self):
        return ''
    def save(self,ofname):
        with open(ofname,'w') as fout:
            fout.write(self.toString())
    def load(self,ifname):
        with open(ifname) as fin:
           self.fromString(fin.read())